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
 * particle back to CPU. At 1e5–1e6 particles that copy (plus per-particle
 * `fillRect`) is the whole frame budget. This class removes BOTH costs: the
 * positions never leave the GPU, and drawing is a single instanced/POINTS draw
 * call.
 *
 * HOW IT AVOIDS THE COPY
 * ----------------------
 * tfjs on the `'webgl'` backend keeps every tensor as a floating-point
 * WebGLTexture. `tensor.dataToGPU()` hands us that texture directly:
 *
 *     { tensorRef: tf.Tensor, texture: WebGLTexture, texShape: [h, w] }
 *
 * A vertex shader then reads particle i straight out of that texture with
 * `texelFetch` (indexed by `gl_VertexID`) and emits a clip-space point. No
 * `arraySync`, no CPU buffer upload, no per-particle JS.
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
 * here — `render()` throws a clear error if the backend is not `'webgl'`.
 * TODO(webgpu): add GpuPointRendererWebGPU reading GPUData.buffer.
 *
 * TEXTURE LAYOUT NOTE
 * -------------------
 * tfjs stores float tensors either UNPACKED (1 float per texel, R channel) or
 * PACKED (tfjs's 2×2-block RGBA scheme). The 2×2 packed layout is NOT a simple
 * row-major dense buffer and this shader does NOT decode it. We infer the
 * stride (floats-per-texel) at runtime from `size / (texW*texH)` and index a
 * DENSE layout. If particles render scrambled, force the simple layout with
 * `tf.env().set('WEBGL_PACK', false)` in the integrator so position/velocity
 * tensors come back unpacked (stride 1). See `stride` computation in `render()`.
 */

import * as tf from "@tensorflow/tfjs";
import * as twgl from "twgl.js";

/** Construction options for {@link GpuPointRenderer}. */
export interface GpuPointOpts {
  /** Point diameter in pixels (clamped to GL's ALIASED_POINT_SIZE_RANGE). Default 2. */
  pointSize?: number;
  /** Clear color as 0–255 RGB, matching ArtPieceConfig.backgroundColor. Default [2,0,12]. */
  background?: [number, number, number];
  /** Speed (px/frame) that maps to the "hot" end of the velocity colormap. Default 4. */
  maxSpeed?: number;
}

const VERT = /* glsl */ `#version 300 es
precision highp float;

// Position texture straight from tf.Tensor.dataToGPU() (never leaves the GPU).
uniform sampler2D u_posTex;
uniform sampler2D u_velTex;
uniform int  u_posTexW;   // width of pos texture in texels
uniform int  u_velTexW;   // width of vel texture in texels
uniform int  u_posStride; // floats per texel in pos tex (1 unpacked / 4 packed-dense)
uniform int  u_velStride;
uniform int  u_hasVel;    // 0/1
uniform vec2 u_resolution;// canvas [w,h] in pixels
uniform float u_pointSize;
uniform float u_maxSpeed;

out vec3 v_color;

// Fetch float at flat index \`flat\` from a dense float texture.
float fetchFloat(sampler2D tex, int texW, int stride, int flatIdx) {
  int texelIndex = flatIdx / stride;
  int channel    = flatIdx - texelIndex * stride; // flatIdx % stride
  ivec2 coord    = ivec2(texelIndex % texW, texelIndex / texW);
  vec4 texel     = texelFetch(tex, coord, 0);
  return texel[channel]; // dynamic vec index is legal in GLSL ES 3.00
}

void main() {
  int i = gl_VertexID;          // one vertex == one particle, no CPU buffer
  float px = fetchFloat(u_posTex, u_posTexW, u_posStride, i * 2);
  float py = fetchFloat(u_posTex, u_posTexW, u_posStride, i * 2 + 1);

  // Pixel space (origin top-left) -> clip space (origin center, y up).
  vec2 clip = (vec2(px, py) / u_resolution) * 2.0 - 1.0;
  clip.y = -clip.y;
  gl_Position = vec4(clip, 0.0, 1.0);
  gl_PointSize = u_pointSize;

  // Optional velocity -> color (cool = slow, hot = fast).
  vec3 col = vec3(0.55, 0.72, 1.0); // default cool blue-white
  if (u_hasVel == 1) {
    float vx = fetchFloat(u_velTex, u_velTexW, u_velStride, i * 2);
    float vy = fetchFloat(u_velTex, u_velTexW, u_velStride, i * 2 + 1);
    float t = clamp(length(vec2(vx, vy)) / u_maxSpeed, 0.0, 1.0);
    col = mix(vec3(0.25, 0.55, 1.0), vec3(1.0, 0.55, 0.2), t); // blue -> orange
  }
  v_color = col;
}
`;

const FRAG = /* glsl */ `#version 300 es
precision highp float;
in vec3 v_color;
out vec4 outColor;
void main() {
  // Round the square GL point into a soft disc.
  vec2 d = gl_PointCoord - vec2(0.5);
  float r2 = dot(d, d);
  if (r2 > 0.25) discard;
  float alpha = smoothstep(0.25, 0.02, r2); // soft edge falloff
  outColor = vec4(v_color * alpha, alpha);
}
`;

/**
 * Zero-copy WebGL2 point renderer for tfjs position tensors.
 *
 * Draws into tfjs's own WebGL2 context (see class docs) so that textures from
 * `dataToGPU()` are usable without any cross-context copy. Additive blending
 * gives particles a glow; the framebuffer is cleared to `background` each frame
 * (no trails — trail/ghost compositing is the Canvas2D renderers' job and is
 * out of scope for this perf lane).
 */
export class GpuPointRenderer {
  private readonly canvas: HTMLCanvasElement;
  private readonly pointSize: number;
  private readonly bg: [number, number, number];
  private readonly maxSpeed: number;

  private gl: WebGL2RenderingContext | null = null;
  private programInfo: twgl.ProgramInfo | null = null;
  private vao: WebGLVertexArrayObject | null = null;

  constructor(canvas: HTMLCanvasElement, opts: GpuPointOpts = {}) {
    this.canvas = canvas;
    this.pointSize = opts.pointSize ?? 2;
    this.bg = opts.background ?? [2, 0, 12];
    this.maxSpeed = opts.maxSpeed ?? 4;
  }

  /**
   * Register `canvas`'s WebGL2 context as the context tfjs will render into.
   * MUST be called BEFORE the tfjs webgl backend is initialized
   * (`tf.setBackend('webgl')` / `tf.ready()`), so that tfjs textures and this
   * renderer share one context and one on-screen surface.
   *
   * @returns the WebGL2 context now owned by both tfjs and this renderer.
   */
  static registerCanvasWithTf(
    canvas: HTMLCanvasElement
  ): WebGL2RenderingContext {
    const gl = canvas.getContext("webgl2", {
      antialias: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
    }) as WebGL2RenderingContext | null;
    if (!gl) throw new Error("GpuPointRenderer: WebGL2 unavailable");
    // Float render/read targets used by tfjs and by texelFetch on float tex.
    gl.getExtension("EXT_color_buffer_float");
    gl.getExtension("OES_texture_float_linear");
    // Hand this exact context to tfjs so dataToGPU() textures live here.
    (tf as any).webgl.setWebGLContext(2, gl);
    return gl;
  }

  /**
   * Draw `posTensor` ([N,2] pixel coords) as N points, reading positions
   * directly from GPU memory. If `velTensor` ([N,2]) is provided, points are
   * colored by speed. No `arraySync`, no CPU copy.
   *
   * @param posTensor [N,2] particle positions in pixels (must be on 'webgl' backend)
   * @param velTensor [N,2] particle velocities, or null to skip velocity color
   * @param w canvas width in pixels
   * @param h canvas height in pixels
   * @param frame frame counter (reserved for future effects; unused today)
   */
  render(
    posTensor: tf.Tensor2D,
    velTensor: tf.Tensor2D | null,
    w: number,
    h: number,
    _frame: number
  ): void {
    if (tf.getBackend() !== "webgl") {
      throw new Error(
        "GpuPointRenderer requires tf.getBackend()==='webgl' for the " +
          "dataToGPU texture path (webgpu GPUBuffer path is a TODO)."
      );
    }
    const gl = this.ensureGl();
    const pi = this.programInfo!;

    if (this.canvas.width !== w) this.canvas.width = w;
    if (this.canvas.height !== h) this.canvas.height = h;
    gl.viewport(0, 0, w, h);

    // Clean slate + additive glow. (Trails are out of scope — see class docs.)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.disable(gl.DEPTH_TEST);
    gl.clearColor(this.bg[0] / 255, this.bg[1] / 255, this.bg[2] / 255, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

    const n = posTensor.shape[0];

    // Pull the GPU-resident texture handles. The returned tensorRef must be
    // disposed after we've issued the draw (originals stay alive in main.ts,
    // so disposing these clones does NOT delete the underlying textures).
    const posGpu = posTensor.dataToGPU();
    const velGpu = velTensor ? velTensor.dataToGPU() : null;

    const posLayout = layoutOf(posGpu, posTensor.size);
    const velLayout = velGpu ? layoutOf(velGpu, velTensor!.size) : null;

    gl.useProgram(pi.program);
    gl.bindVertexArray(this.vao); // empty VAO; shader uses gl_VertexID only
    twgl.setUniforms(pi, {
      u_posTex: posGpu.texture,
      u_posTexW: posLayout.texW,
      u_posStride: posLayout.stride,
      u_velTex: velGpu ? velGpu.texture : posGpu.texture, // sampler must bind
      u_velTexW: velLayout ? velLayout.texW : 1,
      u_velStride: velLayout ? velLayout.stride : 1,
      u_hasVel: velGpu ? 1 : 0,
      u_resolution: [w, h],
      u_pointSize: this.pointSize,
      u_maxSpeed: this.maxSpeed,
    });

    gl.drawArrays(gl.POINTS, 0, n);

    gl.bindVertexArray(null);
    posGpu.tensorRef.dispose();
    velGpu?.tensorRef.dispose();
  }

  /** Release GL resources. Does NOT touch tfjs's shared context. */
  destroy(): void {
    const gl = this.gl;
    if (!gl) return;
    if (this.programInfo) gl.deleteProgram(this.programInfo.program);
    if (this.vao) gl.deleteVertexArray(this.vao);
    this.programInfo = null;
    this.vao = null;
    this.gl = null;
  }

  /** Lazily grab tfjs's GL context (backend must be ready) and build program. */
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
    this.programInfo = twgl.createProgramInfo(gl, [VERT, FRAG]);
    this.vao = gl.createVertexArray();
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
  const [texH, texW] = texShape;
  const texels = Math.max(1, texH * texW);
  const stride = Math.max(1, Math.round(size / texels)); // 1 or 4 in practice
  return { texW, stride };
}
