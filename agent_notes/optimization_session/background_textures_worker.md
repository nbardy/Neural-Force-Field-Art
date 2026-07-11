# Procedural Background Texture Worker

Added a standalone WebGPU background generator for the 3D splat raster. No
existing raster or optimizer files were changed.

## API

```ts
const background = await BackgroundTextureGenerator.create(device, {
  H,
  W,
  mode: "blurred_noise",
  seed: 7,
});

const encoder = device.createCommandEncoder();
background.recordGenerate(encoder, step, strength);
device.queue.submit([encoder.finish()]);

background.buffer; // GPUBuffer containing planar f32 RGB: R[H*W], G[H*W], B[H*W]
background.destroy();
```

Modes are `black`, `dark_solid`, `blurred_noise`, `checkerboard`, and
`fourier`. The mode and dimensions are fixed at creation. `seed` and `step` are
unsigned 32-bit integers. `strength` is clamped to `[0, 1]` and linearly fades
all non-black modes from black to their generated texture.

The output buffer has `STORAGE | COPY_SRC | COPY_DST` usage. Generation only
records commands; the caller owns command submission. Since the parameter
uniform is updated with `queue.writeBuffer`, record one generation per generator
per queue submission when using different step or strength values.

## Shader

The WGSL is self-contained in `background_textures.ts` and writes one RGB pixel
per invocation using an 8x8 workgroup. All modes use the same planar storage
contract:

- `black`: exact zero.
- `dark_solid`: one deterministic dark RGB value for the frame.
- `blurred_noise`: three octaves of smooth value noise with correlated color.
- `checkerboard`: two deterministic dark colors with a seeded offset.
- `fourier`: six seeded sinusoidal components with correlated color fields.

## Verification

Run:

```bash
bun tools/splat3d/background_textures_test.ts
```

The smoke test uses non-workgroup-aligned dimensions, reads the storage buffer
back from WebGPU, prints per-mode range/mean/variance/neighbor statistics, and
checks bounds, finite values, planar layout, same-step determinism, step and seed
variation, mode-specific structure, and linear strength scaling.
