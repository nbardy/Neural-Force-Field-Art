# Feature32 Colorizer Worker

## Delivered Contract

`src/splat3d_feature/` is a standalone WebGPU residual colorizer for a batched,
channel-planar 32D feature image:

```text
residual[o,p] = bias[o] + sum_c weights[o,c] * feature[c,p]
logit[o,p]    = feature[o,p] + residualScale * residual[o,p]
rgb[o,p]      = sigmoid(logit[o,p])
```

The default residual scale is `0.1`. Weights (`3 x 32`) and bias (`vec4`, RGB
in xyz) are explicitly zeroed at creation, so initial output is exactly the
sigmoid of feature planes 0-2. The first three planes therefore remain the RGB
logit skip and channels 3-31 cannot perturb initial RGB.

Images use `[batch][channel][pixel]` f32 planar layout. Pixel planes are viewed
as `vec4<f32>` groups, requiring `width * height` to be divisible by four.
Weights and weight gradients are `[rgb][featureGroup] vec4<f32>`, where each of
the eight feature groups holds four adjacent feature channels.

## Runtime API

- `Feature32Colorizer.create(device, config)` compiles forward, feature-gradient,
  and parameter-gradient pipelines and owns `weights`, `bias`, `weightGrad`, and
  `biasGrad` storage buffers.
- `createIOState(...)` binds external feature/rgb/gradient buffers or aligned
  slices. This is the insertion point for feature-raster output and CLIP input.
- `createOwnedIO()` is a test/prototyping convenience that allocates all four IO
  buffers.
- `recordForward(encoder, io)` writes decoded RGB.
- `recordBackward(encoder, io)` overwrites feature, weight, and bias gradients.
  The parameter reduction covers the configured batch, so no float atomics or
  per-lane host reduction is needed.
- `setParameters(...)`, `zeroParameters()`, and `zeroParameterGradients()` expose
  optimizer/upload control without coupling the module to Adam.

Backward uses the already-produced RGB values to apply the sigmoid derivative.
Feature gradients are one dispatch over eight feature groups. Parameter
gradients use eight workgroups, each reducing four adjacent input channels for
all three RGB rows with workgroup `vec4` reductions.

## Integration Needs

The feature raster still needs to provide contiguous batch-major
`32 * H * W` output and gradient buffers. Wire the colorizer RGB output directly
to the CLIP input buffer, then order one command encoder as:

```text
feature raster forward -> colorizer forward -> CLIP forward/backward
-> colorizer backward -> feature raster backward -> parameter optimizers
```

The colorizer parameter gradients are overwritten per `recordBackward`, not
added. One colorizer configured for the full CLIP batch should be used when the
decoder is shared across views. An optimizer for the 96 weights and three live
bias lanes remains to be connected by the integration owner.

## Verification

`bun tools/splat3d/feature_colorizer_test.ts` runs the emitted WGSL through
`bun-webgpu`. It checks zero-residual RGB parity, nonzero forward parity, and
central finite differences for every feature, weight, and live bias gradient.
