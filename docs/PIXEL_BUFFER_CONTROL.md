# Direct Pixel Buffer Control

`pixel buffer` in `splat.html` is a deliberately unstructured control for the
2D prompt optimizer. It owns one trainable RGB logit per 256x256 pixel and
uses the same `VisionTrainer` as RGB splats and Feature8:

```text
pixel logits -> sigmoid image -> MobileCLIP forward/loss/backward -> sigmoid VJP -> Adam
```

There is no copied CLIP implementation and no CPU image round trip. The image
is copied directly into the trainer's input slot and CLIP's pixel gradient is
copied directly back into the pixel chain.

## Why It Exists

It separates two hypotheses:

1. If direct pixels and splats both fail, the frozen CLIP objective or prompt
   schedule is the limiting factor.
2. If pixels improve cosine while looking like texture, the objective needs a
   prior or augmentation rather than more optimizer momentum.
3. If splats lag pixels in both cosine and image quality, the splat parameter
   path is the limiting factor.

On the current MobileCLIP cat fixture, the raw pixel control reached cosine
`0.16453 -> 0.44585` in 40 steps, but visually became high-frequency texture.
That is expected: direct pixels have no spatial, compositional, or natural-image
prior. It should remain an ablation and a testbed for later low-pass/cutout
experiments, not replace the splat renderer as the default.

## Verification

```bash
bun tools/splat/pixel_optimize_test.ts 40
```

The gate writes `/tmp/pixel_buffer_after.png`, reports cosine improvement, and
uses the production MobileCLIP train plan/weights.
