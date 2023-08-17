# TrashPanda JS

<img width="300" src="out-0.png"></img>

Trash Panda JS is a library for writing modern architectures in tensorflow.js

It is a collection of layers, blocks, and models re-implemented in tfjs. Hopefully, Some of them will
be upstreamed to @tfjs/layers and other will remain research.

Why?
I tried to impliment the famous decoder only GPT in tfjs and I noticed there was a lot missing in the standard lib that you would find in torch or tensorflow.

Not even attention was available.

I Decided why not start implimenting layers here. This will become my playground for implementing papers. And I can test the different layers visually with generative art to give particularly interesting renditions of he rapidly advancing field of attention mechanisms. I will pull out the common std lib ML code that is in keras and pytorch I will need to port.

Also the particle simulator should give a particularly interesting style of visualization showing the different inductive biases visually in the patterns.

Goals:

- [x] MHSA
- [x] Transformer
  - [x] Decoder only Self Atttention
- [x] Rotary Embeddings
- [x] Max Embeddings(Experimental Point Embeddings)
- [ ] Local attention
- [ ] Dilated Global Residual (My idea for a simple speedup)
  - [ ] Windowed Attention
- [x] Shfited attention(warning not tested)
  - [x] Windowed Attention
  - [x] Shifted Windows
  - [x[ Stacked Shifted Window Blocks
- [ ] Mobile Attention (Depthwise Separable Convolutions) (MobileVIT)
- [ ] Dilated Attention
- [ ] Memory Aware Attention(Implement it naively from the torch repo)
  - [ ] Re-Implement with tfjs specific memory management (will need to benchmark with WEBGPU and WebGL)

Stuff Done:

Geometric Embeddings

- [x] sphericalRotationEmbedding
- [x] hypersphereRotationEmbedding

```
import * as tp from "./trashPanda"

tp.linalg.normalize
tp.models.Transformers
tp.models.layers.MultiheadAttention
```
