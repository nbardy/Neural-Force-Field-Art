/**
 *
 * An artistic generative art library that runs entirely on the GPU
 *
 * Why?
 *
 * 1. Drawing counts in the 1,000,000s should be available to generative artists who know only
 *    drawTriangle, drawCircle.
 *
 *    Right now generative artists need to be able to write shaders to draw with WebGL. This will
 *    provide access to that with a high level API familiar to them.
 *
 * 2. I want to be able to render from large tfjs tensors that are holding state and
 *    being computed on by tfjs. Reusing another API would require copying the data to CPU
 *    or doing compute on the CPU. Computing the geometry on the GPU is much faster.
 *
 *    Provides one simple function for passing data back and forth between tfjs
 *    and into shaders, which allows custom rendering build on top of tfjs tensors
 *    for those want no limitations.
 *
 * How?
 *
 * WebGL has no geometry shaders so we do the geometry calculations on GPU with tfjs.
 * Other shader libraries require you to create CPU object and do geometry calculations
 * on the CPU which are slow for computation and causes a lot of data transfer between CPU
 * and GPU.
 *
 * We also accept tfjs tensors directly as input to the drawing functions. This means you can
 * compute your update code with tfjs and draw the results directly without copying to CPU. Instead
 * of onlying using the GPU for drawing you can use it for computation as well. We want
 * our scenes to both render fast and update fast.
 *
 * Consists of:
 *  1. One helper function (drawTWGL that draws TWGIL with default gl.TRIANGLES) or another shape.
 *  2. Artistic Layer that provides drawing oriented APIs instead of GPU oriented APIs
 *     e.g. Position, Direction, etc... over  triangle vertices
 *
 * Starting as a simple wrapper over twgl.js
 *
 * Example wave
 *
 * // range
 * const x = tf.linspace(-1, 1, 100);
 * let y = tf.sin(x);
 *
 * // loop and draw updating positions with sin wave
 * while (true) {
 *    const t = tf.scalar(Date.now() * 0.001);
 *
 *    // move up and down
 *    y = y + tf.sin(x + t);
 *
 *    // draw circle
 *    drawCircles(canvas, x, y, 0.01);
 * }
 *
 */
import * as tf from "@tensorflow/tfjs";
import { buffer, GPUData, input } from "@tensorflow/tfjs";
import * as twgl from "twgl.js";
import { normalize } from "./trashPanda/linalg";

type NumberInput = tf.Tensor | number;
type Color = [number, number, number, number];
type ColorInput = tf.Tensor | Color;

// Stored shaders
const shaderCache = new Map<string, any>();

const bgLibString = `
  vec4 computeColorBackground(vec4 color) {
    return color;
  }

  vec4 computeLinearGradient(vec2 position, vec4 startColor, vec4 endColor, float angle) {
    float t = dot(position, vec2(cos(angle), sin(angle)));
    return mix(startColor, endColor, t);
  }

  vec4 computeRadialGradient(vec2 position, vec4 startColor, vec4 endColor, float radius) {
    float t = length(position) / radius;
    return mix(startColor, endColor, t);
  }

  vec4 getBackground(int type, vec2 position, vec4 color, vec4 startColor, vec4 endColor, float angle, float radius) {
    if (type == 0) return computeColorBackground(color);
    if (type == 1) return computeLinearGradient(position, startColor, endColor, angle);
    if (type == 2) return computeRadialGradient(position, startColor, endColor, radius);
    // Other types can be handled here
    return vec4(1.0, 1.0, 1.0, 1.0); // Default white background
  }
`;

/**
 *  Converts a uniform input in js to it's GLSL type
 */
const uniformToStringLine = (key: string, value: any) => {
  // Converts a JS type to a GLSL type
  const dtype = (input: any) => {
    // is int
    if (Number.isInteger(input)) {
      return "int";
    } else if (typeof input === "number") {
      return "float";
    } else if (typeof input === "boolean") {
      return "bool";
    } else {
      throw new Error("Invalid type");
    }
  };

  if (value instanceof tf.Tensor) {
    // check if vec2, vec3, vec4
    const shape = value.shape;
    const B = shape[0];
    const D = shape[1];
    // assert
    const isSize = (s: number) => s === 2 || s === 3 || s === 4;
    if (!isSize(D)) {
      console.log("Invalid shape", D);
      throw new Error("Invalid shape");
    }

    const v = `vec${shape[1]}`;

    return `uniform ${v} ${key};`;
  } else {
    return `uniform ${dtype(value)} ${key};`;
  }
};

function getCompiledShader(gl, shaderSource: string) {
  if (shaderCache.has(shaderSource)) {
    return shaderCache.get(shaderSource);
  }

  const programInfo = twgl.createProgramInfo(gl, [shaderSource]);
  shaderCache.set(shaderSource, programInfo);
  return programInfo;
}

export interface BackgroundConfig {
  type: "color" | "linear-gradient" | "radial-gradient" | "custom";
  color?: number[] | tf.Tensor;
  startColor?: number[] | tf.Tensor;
  endColor?: number[] | tf.Tensor;
  angle?: number;
  radius?: number;
}

export function drawTWGL(
  gl,
  pos: tf.Tensor,
  shape: number,
  fragmentShader?: string,
  uniforms?: object,
  backgroundConfig?: BackgroundConfig
) {
  const backgroundType = backgroundConfig?.type || "color";
  let bgTypeUniform: number;
  if (backgroundType === "color") bgTypeUniform = 0;
  else if (backgroundType === "linear-gradient") bgTypeUniform = 1;
  else if (backgroundType === "radial-gradient") bgTypeUniform = 2;
  else bgTypeUniform = 0; // Default to color

  uniforms = uniforms || {};

  const posBuffer = extractBufferFromTexture(gl, pos);

  const bufferInfo: twgl.BufferInfo = {
    numElements: shape, // You might need to determine the correct value for numElements
    attribs: {
      a_position: {
        numComponents: 2, // or 3 or 4 depending on your data
        buffer: posBuffer,
      },
    },
  };

  // get keys of default uniforms
  const inputUniforms = Object.keys(uniforms);
  const inputUniformsString = inputUniforms
    .map((key) => {
      return uniformToStringLine(key, inputUniforms[key]);
    })
    .join("\n");

  // Additional uniforms for background
  uniforms["u_backgroundType"] = bgTypeUniform;
  uniforms["u_bgColor"] = backgroundConfig?.color || [1, 1, 1, 1];
  uniforms["u_bgStartColor"] = backgroundConfig?.startColor || [1, 1, 1, 1];
  uniforms["u_bgEndColor"] = backgroundConfig?.endColor || [1, 1, 1, 1];
  uniforms["u_bgAngle"] = backgroundConfig?.angle || 0;
  uniforms["u_bgRadius"] = backgroundConfig?.radius || 1;

  // Custom fragment shader with background functions included
  const customFragmentShader = `${bgLibString}
    precision mediump float;
    varying vec4 v_color;
    uniform int u_backgroundType;
    uniform vec4 u_bgColor;
    uniform vec4 u_bgStartColor;
    uniform vec4 u_bgEndColor;
    uniform float u_bgAngle;
    uniform float u_bgRadius;

    ${inputUniformsString}

    void main() {
      vec4 bgColor = getBackground(u_backgroundType, gl_FragCoord.xy, u_bgColor, u_bgStartColor, u_bgEndColor, u_bgAngle, u_bgRadius);
      gl_FragColor = bgColor;
    }
  `;

  // Create program with custom or default shader
  const programInfo = getCompiledShader(
    gl,
    fragmentShader || customFragmentShader
  );

  // Draw the shape
  twgl.setBuffersAndAttributes(gl, programInfo, bufferInfo);
  twgl.setUniforms(programInfo, uniforms);
  twgl.drawBufferInfo(gl, bufferInfo);
}

function computeTriangleVertices(
  pos: tf.Tensor,
  dir: tf.Tensor,
  height: tf.Tensor | number,
  baseWidth: tf.Tensor | number
): tf.Tensor {
  const pos2D = pos.reshape([-1, 2]); // Bx2
  const dir2D = dir.reshape([-1, 2]); // Bx2

  const normalizedDir = normalize(dir2D);
  const perpendicularDir = tf.tensor2d([
    -normalizedDir.arraySync()[0][1],
    normalizedDir.arraySync()[0][0],
  ]);

  const halfBaseWidth =
    baseWidth instanceof tf.Tensor ? baseWidth.div(2) : baseWidth / 2; // B

  const vertex1 = pos2D.add(normalizedDir.mul(height));
  const vertex2 = pos2D.add(perpendicularDir.mul(halfBaseWidth));
  const vertex3 = pos2D.sub(perpendicularDir.mul(halfBaseWidth));

  return tf.stack([vertex1, vertex2, vertex3], 1).reshape([-1, 6]); // Bx6
}

export function drawCircles(
  canvas,
  positions: tf.Tensor,
  radius: NumberInput,
  fragmentShader?: string,
  background?: { color?: ColorInput; image?: string }
) {
  const gl = canvas.getContext("webgl");

  // Generate uniforms
  const uniforms = {
    u_radius: radius,
    u_color: background?.color || [1, 1, 1, 1],
    u_image: background?.image || null,
  };

  // Custom fragment shader for circle
  const circleFragmentShader =
    fragmentShader ||
    `
    precision mediump float;
    uniform float u_radius;
    uniform vec4 u_color;
    void main() {
      float distance = length(gl_PointCoord - vec2(0.5));
      if (distance < u_radius) {
        gl_FragColor = u_color;
      } else {
        discard;
      }
    }
  `;

  // Draw circles using TWGL
  drawTWGL(gl, positions, gl.POINTS, circleFragmentShader, uniforms);
}

type ColorAll = tf.Tensor | number[];

/**
 *  Since we accept all color types, we need to normalize them to a tensor
 *  to make the code the same for all color types.
 */
export const normalizeColorType = (color: ColorAll) => {
  if (color instanceof tf.Tensor) {
    return color;
  } else if (Array.isArray(color)) {
    return tf.tensor1d(color);
  } else {
    throw new Error("Invalid color type");
  }
};

export function drawTriangles(args: {
  canvas: HTMLCanvasElement;
  pos: tf.Tensor; // Bx2
  dir?: tf.Tensor; // Bx2 | Length 2 Vector
  height?: NumberInput; // B | Scalar
  baseWidth?: tf.Tensor; // B | Scalar
  fragmentShader?: string;
  color?: ColorAll; // Bx4 | Length 4 Vector
}) {
  const argsWithDefault = {
    ...args,
    dir: args.dir ?? tf.tensor2d([0, 1]),
    height: args.height ?? tf.tensor([10]),
    baseWidth: args.baseWidth ?? tf.tensor([10]),
    color: args.color && normalizeColorType(args.color),
  };

  const { canvas, pos, dir, height, baseWidth, fragmentShader, color } =
    argsWithDefault;
  // Compute triangle vertices using pos, dir, height, and baseWidth

  // if dir is empty. make vertical
  // let dirWithDefault = dir ?? tf.tensor2d([0, 1]);
  // cleaner

  const vertices = computeTriangleVertices(pos, dir, height, baseWidth); // Bx6

  const gl = canvas.getContext("webgl");
  if (!gl) {
    throw new Error("WebGL not supported");
  }

  // Generate uniforms
  const uniforms = {
    u_color: color,
    // Add pos, dir, height, and baseWidth as uniforms (TODO: Can uniforms be tensors?)
    u_pos: pos,
    u_dir: dir,
    u_height: height,
    u_baseWidth: baseWidth,
  };

  // Draw triangles using TWGL
  drawTWGL(gl, vertices, gl.TRIANGLES, fragmentShader, uniforms);
}

// Webgl rendering context extended with pixel pack buffer
export interface WebGLContextElite extends WebGLRenderingContext {
  PIXEL_PACK_BUFFER: number;
  STATIC_READ: number;

  // WEBGL 2.0
  // TODO: Fix this type
  getBufferSubData: any;
}

// This allows us to
export function extractBufferFromTexture(
  glInput: WebGLRenderingContext,
  tensor: tf.Tensor
): WebGLBuffer {
  // Type Fix
  // Types are old ( can remove this when types are updated)
  let gl = glInput as WebGLContextElite;

  const gpuData = tensor.dataToGPU();

  if (gpuData.texture == null || gpuData.texShape == null) {
    throw new Error("GPUData texture or texShape is null");
  }

  // Create a framebuffer and bind it
  const framebuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);

  // Attach the texture to the framebuffer
  gl.framebufferTexture2D(
    gl.FRAMEBUFFER,
    gl.COLOR_ATTACHMENT0,
    gl.TEXTURE_2D,
    gpuData.texture,
    0
  );

  // Check if the framebuffer is complete
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
    throw new Error("Framebuffer is not complete");
  }

  // Create a buffer and bind it
  const buffer = gl.createBuffer();

  if (!buffer) {
    throw new Error("Buffer is null");
  }

  gl.bindBuffer(gl.PIXEL_PACK_BUFFER, buffer);
  gl.bufferData(
    gl.PIXEL_PACK_BUFFER,
    gpuData.texShape[0] * gpuData.texShape[1] * 4,
    gl.STATIC_READ
  );

  // Read the pixels from the framebuffer into the buffer
  gl.readPixels(
    0,
    0,
    gpuData.texShape[1],
    gpuData.texShape[0],
    gl.RGBA,
    gl.FLOAT,
    null // TODO: Should this be different
  );

  // Unbind the framebuffer and the buffer
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);

  return buffer;
}
