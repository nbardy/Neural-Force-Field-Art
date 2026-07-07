/**
 * microgpu — a ~1-screen zero-dependency WebGPU helper.
 *
 * NOT an engine. It just collapses the one-time WebGPU boilerplate (canvas
 * config, render pipeline from one WGSL module, uniform buffer, bind group, and
 * a single render pass) into a handful of typed calls. Everything else stays raw
 * WebGPU so there's no abstraction to fight. See src/render/webgpu/points.ts for
 * a user.
 *
 * Device-SHARING is the whole point: we attach the canvas to an EXISTING
 * GPUDevice (tfjs's, via tf.backend().device) so a GPUBuffer from
 * tensor.dataToGPU() can be bound directly — zero copy, no readback.
 */

export interface GpuCtx {
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
}

/** Additive blend (glow): out = src.rgb*src.a + dst. */
export const BLEND_ADD: GPUBlendState = {
  color: { srcFactor: "src-alpha", dstFactor: "one", operation: "add" },
  alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
};

/** Attach `canvas` to an already-created `device` (e.g. tfjs's webgpu device). */
export function attachCanvas(
  canvas: HTMLCanvasElement,
  device: GPUDevice
): GpuCtx {
  const context = canvas.getContext("webgpu") as GPUCanvasContext | null;
  if (!context) {
    throw new Error("microgpu: canvas.getContext('webgpu') returned null");
  }
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });
  return { device, context, format };
}

/** Build a render pipeline from ONE WGSL module with `vs`/`fs` entry points. */
export function renderPipeline(
  device: GPUDevice,
  opts: {
    code: string;
    format: GPUTextureFormat;
    blend?: GPUBlendState;
    topology?: GPUPrimitiveTopology;
  }
): GPURenderPipeline {
  const module = device.createShaderModule({ code: opts.code });
  return device.createRenderPipeline({
    layout: "auto",
    vertex: { module, entryPoint: "vs" },
    fragment: {
      module,
      entryPoint: "fs",
      targets: [{ format: opts.format, blend: opts.blend }],
    },
    primitive: { topology: opts.topology ?? "triangle-strip" },
  });
}

/** Build a compute pipeline from ONE WGSL module with a `main` entry point. */
export function computePipeline(
  device: GPUDevice,
  code: string
): GPUComputePipeline {
  const module = device.createShaderModule({ code });
  return device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
}

/** A UNIFORM|COPY_DST buffer of `byteLength` bytes (multiple of 4). */
export function uniformBuffer(device: GPUDevice, byteLength: number): GPUBuffer {
  return device.createBuffer({
    size: (byteLength + 3) & ~3,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

/** Bind group against a pipeline's auto layout (group 0). */
export function bindGroup(
  device: GPUDevice,
  pipeline: GPURenderPipeline,
  entries: GPUBindGroupEntry[]
): GPUBindGroup {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });
}

/**
 * Record + submit one render pass to the canvas. `clear` = clear colour, or
 * null to load (preserve) existing contents. `draw` sets pipeline/bindgroup and
 * issues draw calls.
 */
export function renderPass(
  ctx: GpuCtx,
  clear: GPUColor | null,
  draw: (pass: GPURenderPassEncoder) => void
): void {
  const encoder = ctx.device.createCommandEncoder();
  const view = ctx.context.getCurrentTexture().createView();
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view,
        clearValue: clear ?? undefined,
        loadOp: clear ? "clear" : "load",
        storeOp: "store",
      },
    ],
  });
  draw(pass);
  pass.end();
  ctx.device.queue.submit([encoder.finish()]);
}
