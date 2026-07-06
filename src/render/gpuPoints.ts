/**
 * GpuPointRenderer — zero-copy GPU particle rendering (perf lane)
 * ===============================================================
 *
 * WHAT THIS REPLACES
 * ------------------
 * The live loop in `src/main.ts` currently does, every frame:
 *
 *     const posArr = pos.arraySync() as number[][];   // GPU -> CPU stall
 *     const velArr = vel.arraySync() as number[][];   // GPU -> CPU stall
 *     renderer.render(ctx, w, h, posArr, velArr, frame);   // Canvas2D fillRect
 *
 * `arraySync()` blocks the JS thread until the GPU flushes and copies every
 * particle back to CPU. At 1e5 particles that copy (plus per-particle
 * `fillRect`) is the whole frame budget. This class removes BOTH costs: the
 * positions never leave the GPU, and drawing is a single INSTANCED draw call.
 *
 * HOW IT AVOIDS THE COPY
 * ----------------------
 * tfjs on the `'webgl'` backend keeps every tensor as a floating-point
 * WebGLTexture. `tensor.dataToGPU()` hands us that texture directly:
 *
 *     { tensorRef: tf.Tensor, texture: WebGLTexture, texShape: [rows, cols] }
 *
 * A vertex shader then reads particle `i` straight out of that texture with
 * `texelFetch` (indexed by `gl_InstanceID`) and emits an instanced quad. No
 * `arraySync`, no CPU buffer upload, no per-particle JS.
 *
 * DRAW MODEL — INSTANCED QUADS + ROUND FRAG DOT
 * ---------------------------------------------
 * Each particle is one INSTANCE of a 4-vertex unit quad (triangle strip). The
 * vertex shader places the quad centre at the particle's pixel position and
 * sizes it by `pointSize` px; the fragment shader carves a smooth, sub-pixel
 * anti-aliased ROUND dot out of that square:
 *
 *     float d = length(uv - 0.5);
 *     float alpha = smoothstep(0.5, 0.5 - fwidth(d) * 1.5, d);
 *
 * This is preferred over `gl.POINTS` because `gl_PointSize` is clamped to a
 * driver-dependent ALIASED_POINT_SIZE_RANGE (often max 1–64) and has no AA;
 * instanced quads render at any size with clean edges.
 *
 * TRAIL / GHOST FADE
 * ------------------
 * The context is created with `preserveDrawingBuffer: true`, so the previous
 * frame survives. Each frame we first draw a faint full-screen quad of the
 * background colour at `alphaBlend` opacity (normal alpha blend), nudging the
 * whole image toward the background — then stamp the new particles ADDITIVELY
 * on top. Standing particles bloom and saturate; departed ones fade over many
 * frames. This mirrors the Canvas2D `AlphaFadeRenderer` look in `renderers.ts`.
 *
 * ⚠️ CONTEXT-SHARING REQUIREMENT (the load-bearing gotcha)
 * -------------------------------------------------------
 * A `WebGLTexture` belongs to the WebGL context that created it and CANNOT be
 * used from a different context. The texture from `dataToGPU()` lives in
 * *tfjs's* WebGL2 context. Therefore this renderer draws into THAT SAME
 * context (obtained via `tf.backend().gpgpu.gl`) — not into a fresh context we
 * make ourselves. For the pixels to be visible on screen, tfjs must be running
 * on the on-screen canvas. Use the static helper below BEFORE initializing the
 * tfjs backend:
 *
 *     // in the integrator, before tf.setBackend('webgl') / tf.ready():
 *     GpuPointRenderer.registerCanvasWithTf(canvas);
 *     // ... then the normal backend init runs and binds tfjs to `canvas`.
 *     const gpu = new GpuPointRenderer(canvas, { pointSize: 2 });
 *
 * If you skip the helper, tfjs renders to its own offscreen canvas and this
 * class will draw correctly but off-screen (nothing visible).
 *
 * FALLBACK / TODO (webgpu backend)
 * --------------------------------
 * On the `'webgpu'` backend `dataToGPU()` returns a `GPUBuffer` (not a
 * WebGLTexture) and there is no WebGL interop. That path needs a parallel WGSL
 * renderer that binds the GPUBuffer as a storage/vertex buffer. Not implemented
 * here — the constructor throws a clear error if the backend is not `'webgl'`.
 * TODO(webgpu): add GpuPointRendererWebGPU reading GPUData.buffer.
 *
 * TEXTURE LAYOUT NOTE
 * -------------------
 * tfjs stores float tensors either UNPACKED (1 float per texel, R channel) or
 * PACKED (tfjs's 2×2-block RGBA scheme). The 2×2 packed layout is NOT a simple
 * row-major dense buffer and this shader does NOT decode it. We infer the
 * stride (floats-per-texel) at runtime from `size / (texW*texH)` and index a
 * DENSE layout. To guarantee the simple layout, `registerCanvasWithTf` also
 * sets `tf.env().set('WEBGL_PACK', false)` so position/velocity tensors come
 * back unpacked (stride 1). See `layoutOf()` for the inference.
 */

import * as tf from "@tensorflow/tfjs";
import * as twgl from "twgl.js";

/** Construction options for {@link GpuPointRenderer}. */
export interface GpuPointOpts {
  /** Dot diameter in pixels (instanced quad, no driver point-size clamp). Default 2. */
  pointSize?: number;
  /** Background/clear colour as 0–255 RGB, matching ArtPieceConfig.backgroundColor. Default [2,0,12]. */
  background?: [number, number, number];
  /**
   * Per-frame fade opacity of the background quad, matching
   * ArtPieceConfig.alphaBlend. Lower = longer trails. Default 0.06.
   */
  alphaBlend?: number;
  /** Speed (px/frame) that maps to the "hot" end of the velocity colormap. Default 4. */
  maxSpeed?: number;
}

// ---------------------------------------------------------------------------
// Point shaders: one INSTANCE per particle, position read from a GPU texture.
// ---------------------------------------------------------------------------
const POINT_VERT = /* glsl */ `#version 300 es
precision highp float;

// Per-vertex: the four corners of a unit quad, each in [0,1].
in vec2 a_corner;

// Position/velocity textures straight from tf.Tensor.dataToGPU() (never leave GPU).
uniform sampler2D u_posTex;
uniform sampler2D u_velTex;
uniform int  u_posTexW;   // width of pos texture in texels
uniform int  u_velTexW;   // width of vel texture in texels
uniform int  u_posStride; // floats per texel in pos tex (1 unpacked / 4 packed-dense)
uniform int  u_velStride;
uniform int  u_hasVel;    // 0/1
uniform vec2 u_resolution;// canvas [w,h] in pixels
uniform float u_pointSize;// dot diameter in pixels
uniform float u_maxSpeed;

out vec2 v_uv;    // [0,1] across the quad, for the round-dot mask in frag
out vec3 v_color;

// Fetch float at flat index \`flatIdx\` from a dense float texture.
float fetchFloat(sampler2D tex, int texW, int stride, int flatIdx) {
  int texelIndex = flatIdx / stride;
  int channel    = flatIdx - texelIndex * stride; // flatIdx % stride
  ivec2 coord    = ivec2(texelIndex % texW, texelIndex / texW);
  vec4 texel     = texelFetch(tex, coord, 0);
  return texel[channel]; // dynamic vec index is legal in GLSL ES 3.00
}

void main() {
  int i = gl_InstanceID;        // one instance == one particle, no CPU buffer
  float px = fetchFloat(u_posTex, u_posTexW, u_posStride, i * 2);
  float py = fetchFloat(u_posTex, u_posTexW, u_posStride, i * 2 + 1);

  // Particle centre: pixel space (origin top-left) -> clip space (centre, y up).
  vec2 centre = (vec2(px, py) / u_resolution) * 2.0 - 1.0;
  centre.y = -centre.y;

  // Offset this vertex to a quad corner. Symmetric, so the y-flip is irrelevant.
  vec2 corner = a_corner - 0.5;                       // [-0.5, 0.5]
  vec2 offset = corner * u_pointSize / u_resolution * 2.0;
  gl_Position = vec4(centre + offset, 0.0, 1.0);

  v_uv = a_corner;              // pass [0,1] quad coord to fragment

  // Optional velocity -> colour (cool = slow, hot = fast); white when no vel.
  vec3 col = vec3(1.0);         // default white
  if (u_hasVel == 1) {
    float vx = fetchFloat(u_velTex, u_velTexW, u_velStride, i * 2);
    float vy = fetchFloat(u_velTex, u_velTexW, u_velStride, i * 2 + 1);
    float t = clamp(length(vec2(vx, vy)) / u_maxSpeed, 0.0, 1.0);
    col = mix(vec3(0.25, 0.55, 1.0), vec3(1.0, 0.55, 0.2), t); // blue -> orange
  }
  v_color = col;
}
`;

const POINT_FRAG = /* glsl */ `#version 300 es
precision highp float;
in vec2 v_uv;
in vec3 v_color;
out vec4 outColor;
void main() {
  // Carve a smooth, sub-pixel anti-aliased round dot out of the unit quad.
  float d = length(v_uv - 0.5);
  float alpha = smoothstep(0.5, 0.5 - fwidth(d) * 1.5, d);
  if (alpha <= 0.0) discard;
  // Additive blend (SRC_ALPHA, ONE): emit straight colour, weight by alpha.
  outColor = vec4(v_color, alpha);
}
`;

// ---------------------------------------------------------------------------
// Background fade shaders: a single full-screen triangle, solid colour+alpha.
// Uses gl_VertexID + a const array — needs NO vertex attributes.
// ---------------------------------------------------------------------------
const BG_VERT = /* glsl */ `#version 300 es
precision highp float;
void main() {
  // Oversized triangle that covers the whole clip rect.
  vec2 p = vec2(
    (gl_VertexID == 1) ? 3.0 : -1.0,
    (gl_VertexID == 2) ? 3.0 : -1.0
  );
  gl_Position = vec4(p, 0.0, 1.0);
}
`;

const BG_FRAG = /* glsl */ `#version 300 es
precision highp float;
uniform vec4 u_color; // rgb = background (0..1), a = alphaBlend
out vec4 outColor;
void main() { outColor = u_color; }
`;

/**
 * Zero-copy WebGL2 point renderer for tfjs position tensors.
 *
 * Draws into tfjs's own WebGL2 context (see class docs) so that textures from
 * `dataToGPU()` are usable without any cross-context copy. Particles are drawn
 * as anti-aliased round dots with additive blending for glow, over a per-frame
 * background fade quad that produces the trail/ghost look of the Canvas2D
 * `AlphaFadeRenderer`.
 */
export class GpuPointRenderer {
  private readonly canvas: HTMLCanvasElement;
  private readonly pointSize: number;
  private readonly bg: [number, number, number];
  private readonly alphaBlend: number;
  private readonly maxSpeed: number;

  private gl: WebGL2RenderingContext | null = null;
  private pointProgram: twgl.ProgramInfo | null = null;
  private bgProgram: twgl.ProgramInfo | null = null;
  private quadBufferInfo: twgl.BufferInfo | null = null;
  private pointVao: WebGLVertexArrayObject | null = null;
  private emptyVao: WebGLVertexArrayObject | null = null;

  private lastW = -1;
  private lastH = -1;
  private needsBgClear = true; // first frame / after resize: paint opaque bg

  /**
   * @param canvas the on-screen canvas already registered with tfjs via
   *   {@link GpuPointRenderer.registerCanvasWithTf}.
   * @param opts see {@link GpuPointOpts}.
   * @throws if the active tfjs backend is not `'webgl'` (the dataToGPU texture
   *   path requires it; main.ts falls back to Canvas2D on other backends).
   */
  constructor(canvas: HTMLCanvasElement, opts: GpuPointOpts = {}) {
    if (tf.getBackend() !== "webgl") {
      throw new Error(
        "GpuPointRenderer requires the tfjs 'webgl' backend (tf.getBackend() " +
          `=== 'webgl'); got '${tf.getBackend()}'. The zero-copy path reads a ` +
          "dataToGPU() WebGLTexture, which does not exist on the webgpu " +
          "backend. Force webgl (tf.setBackend('webgl')) or use the Canvas2D " +
          "fallback renderer."
      );
    }
    this.canvas = canvas;
    this.pointSize = opts.pointSize ?? 2;
    this.bg = opts.background ?? [2, 0, 12];
    this.alphaBlend = opts.alphaBlend ?? 0.06;
    this.maxSpeed = opts.maxSpeed ?? 4;
  }

  /**
   * Register `canvas`'s WebGL2 context as the context tfjs will render into.
   * MUST be called BEFORE the tfjs webgl backend is initialized
   * (`tf.setBackend('webgl')` / `tf.ready()`), so that tfjs textures and this
   * renderer share one context and one on-screen surface.
   *
   * Also disables `WEBGL_PACK` so tfjs stores tensors as simple unpacked float
   * textures — the dense layout this renderer's shader indexes (see class docs).
   *
   * @returns the WebGL2 context now owned by both tfjs and this renderer.
   */
  /** tfjs 4.x exports setWebGLContext at top level (`tf.setWebGLContext`) via
   *  `export * from '@tensorflow/tfjs-backend-webgl'`. Some builds also expose a
   *  `tf.webgl` namespace. Resolve whichever exists (the old code assumed
   *  `tf.webgl.setWebGLContext`, which is undefined in this build → it threw
   *  AFTER getContext('webgl2') had already committed the canvas, poisoning it). */
  private static resolveSetWebGLContext():
    | ((v: number, gl: WebGLRenderingContext) => void)
    | null {
    const t = tf as any;
    const fn = t.setWebGLContext || (t.webgl && t.webgl.setWebGLContext);
    return typeof fn === "function" ? fn : null;
  }

  /**
   * True iff the GPU path can be attempted WITHOUT risk of poisoning the real
   * canvas: WebGL2 is creatable (probed on a THROWAWAY canvas) and tfjs exposes
   * setWebGLContext. Call this before registerCanvasWithTf so a failed setup
   * never commits — and thus never black-screens — the on-screen canvas.
   */
  static isSupported(): boolean {
    try {
      if (!GpuPointRenderer.resolveSetWebGLContext()) return false;
      const probe = document.createElement("canvas").getContext("webgl2");
      return !!probe;
    } catch (_) {
      return false;
    }
  }

  static registerCanvasWithTf(
    canvas: HTMLCanvasElement
  ): WebGL2RenderingContext {
    // Resolve the tfjs API FIRST, before touching the real canvas, so we never
    // commit it to webgl2 and then throw (the original poisoning bug).
    const setWebGLContext = GpuPointRenderer.resolveSetWebGLContext();
    if (!setWebGLContext) {
      throw new Error("GpuPointRenderer: tfjs setWebGLContext() not available");
    }
    const gl = canvas.getContext("webgl2", {
      alpha: false, // opaque canvas: no page-composite premultiply headaches
      antialias: false, // AA comes from the frag dot, not MSAA
      premultipliedAlpha: false,
      preserveDrawingBuffer: true, // KEEP previous frame -> trail/ghost fade
      depth: false,
      stencil: false,
    }) as WebGL2RenderingContext | null;
    if (!gl) throw new Error("GpuPointRenderer: WebGL2 unavailable");
    // Float render/read targets used by tfjs and by texelFetch on float tex.
    gl.getExtension("EXT_color_buffer_float");
    gl.getExtension("OES_texture_float_linear");
    // Simple unpacked (stride-1) float textures so the shader can index densely.
    (tf as any).env().set("WEBGL_PACK", false);
    // Hand this exact context to tfjs so dataToGPU() textures live here.
    setWebGLContext(2, gl);
    return gl;
  }

  /**
   * Draw `posTensor` ([N,2] pixel coords) as N round dots, reading positions
   * directly from GPU memory. If `velTensor` ([N,2]) is provided, dots are
   * coloured by speed; otherwise white. No `arraySync`, no CPU copy.
   *
   * @param posTensor [N,2] particle positions in pixels (on the 'webgl' backend)
   * @param velTensor [N,2] particle velocities, or null to skip velocity colour
   * @param w canvas width in pixels
   * @param h canvas height in pixels
   * @param _frame frame counter (reserved for future effects; unused today)
   */
  render(
    posTensor: tf.Tensor2D,
    velTensor: tf.Tensor2D | null,
    w: number,
    h: number,
    _frame: number
  ): void {
    const gl = this.ensureGl();

    // Resize resets the drawing buffer to transparent, so re-establish the bg.
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
      this.needsBgClear = true;
    }
    if (w !== this.lastW || h !== this.lastH) {
      this.lastW = w;
      this.lastH = h;
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null); // draw to the on-screen canvas
    gl.viewport(0, 0, w, h);
    gl.disable(gl.DEPTH_TEST);

    // First frame (or post-resize): start from an opaque background.
    if (this.needsBgClear) {
      gl.disable(gl.BLEND);
      gl.clearColor(this.bg[0] / 255, this.bg[1] / 255, this.bg[2] / 255, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      this.needsBgClear = false;
    }

    // (1) TRAIL FADE — normal alpha blend a faint bg quad over the prior frame.
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(this.bgProgram!.program);
    gl.bindVertexArray(this.emptyVao); // bg shader uses gl_VertexID only
    twgl.setUniforms(this.bgProgram!, {
      u_color: [
        this.bg[0] / 255,
        this.bg[1] / 255,
        this.bg[2] / 255,
        this.alphaBlend,
      ],
    });
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // (2) POINTS — additive glow, one instanced quad per particle.
    const n = posTensor.shape[0];

    // Pull the GPU-resident texture handles. tfjs mints a fresh clone tensor
    // (tensorRef) per call; we MUST dispose it after drawing or its texture
    // leaks. Disposing the clone does NOT free the caller's original tensors.
    const posGpu = posTensor.dataToGPU();
    const velGpu = velTensor ? velTensor.dataToGPU() : null;
    const posLayout = layoutOf(posGpu, posTensor.size);
    const velLayout = velGpu ? layoutOf(velGpu, velTensor!.size) : null;

    gl.blendFunc(gl.SRC_ALPHA, gl.ONE); // additive
    gl.useProgram(this.pointProgram!.program);
    gl.bindVertexArray(this.pointVao);
    twgl.setUniforms(this.pointProgram!, {
      u_posTex: posGpu.texture,
      u_posTexW: posLayout.texW,
      u_posStride: posLayout.stride,
      // sampler must always bind a valid texture even when unused:
      u_velTex: velGpu ? velGpu.texture : posGpu.texture,
      u_velTexW: velLayout ? velLayout.texW : 1,
      u_velStride: velLayout ? velLayout.stride : 1,
      u_hasVel: velGpu ? 1 : 0,
      u_resolution: [w, h],
      u_pointSize: this.pointSize,
      u_maxSpeed: this.maxSpeed,
    });
    // 4-vertex triangle-strip quad, N instances, one per particle.
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, n);

    gl.bindVertexArray(null);

    // Release the per-frame GPU handles (see comment above).
    posGpu.tensorRef.dispose();
    velGpu?.tensorRef.dispose();
  }

  /** Release GL resources. Does NOT touch tfjs's shared context. */
  destroy(): void {
    const gl = this.gl;
    if (!gl) return;
    if (this.pointProgram) gl.deleteProgram(this.pointProgram.program);
    if (this.bgProgram) gl.deleteProgram(this.bgProgram.program);
    if (this.pointVao) gl.deleteVertexArray(this.pointVao);
    if (this.emptyVao) gl.deleteVertexArray(this.emptyVao);
    this.pointProgram = null;
    this.bgProgram = null;
    this.quadBufferInfo = null;
    this.pointVao = null;
    this.emptyVao = null;
    this.gl = null;
  }

  /**
   * Lazily grab tfjs's GL context (backend must be ready) and build programs,
   * the unit-quad buffer, and the VAOs. Isolating attribute state in our own
   * VAO keeps tfjs's compute draws from clobbering our quad bindings.
   */
  private ensureGl(): WebGL2RenderingContext {
    if (this.gl) return this.gl;
    const gl = (tf.backend() as any).gpgpu?.gl as
      | WebGL2RenderingContext
      | undefined;
    if (!gl) {
      throw new Error(
        "GpuPointRenderer: could not obtain tfjs WebGL context. Ensure the " +
          "'webgl' backend is initialized (await tf.ready())."
      );
    }
    this.gl = gl;
    this.pointProgram = twgl.createProgramInfo(gl, [POINT_VERT, POINT_FRAG]);
    this.bgProgram = twgl.createProgramInfo(gl, [BG_VERT, BG_FRAG]);

    // Unit quad as a triangle strip: (0,0) (1,0) (0,1) (1,1).
    this.quadBufferInfo = twgl.createBufferInfoFromArrays(gl, {
      a_corner: { numComponents: 2, data: [0, 0, 1, 0, 0, 1, 1, 1] },
    });

    // Record the quad attribute state once into a dedicated VAO.
    this.pointVao = gl.createVertexArray();
    gl.bindVertexArray(this.pointVao);
    twgl.setBuffersAndAttributes(gl, this.pointProgram, this.quadBufferInfo);
    gl.bindVertexArray(null);

    // Empty VAO for the attribute-less background pass.
    this.emptyVao = gl.createVertexArray();

    return gl;
  }
}

/**
 * Infer the dense float-texture layout of a dataToGPU() result.
 * stride = floats packed per texel (1 for unpacked R32F, 4 for dense RGBA).
 * See "TEXTURE LAYOUT NOTE" in the class docs for the packed-layout caveat.
 */
function layoutOf(
  gpu: { texShape?: [number, number] },
  size: number
): { texW: number; stride: number } {
  const texShape = gpu.texShape;
  if (!texShape) {
    // No shape reported — assume a single dense RGBA row.
    return { texW: Math.ceil(size / 4), stride: 4 };
  }
  const [texH, texW] = texShape; // tfjs reports [rows, cols] == [height, width]
  const texels = Math.max(1, texH * texW);
  const stride = Math.max(1, Math.round(size / texels)); // 1 or 4 in practice
  return { texW, stride };
}
