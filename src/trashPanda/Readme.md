# TrashPanda JS

<img width="300" src="out-0.png"></img>

Trash Panda JS is a library for writing modern attention architectures in tensorflow.js

It is a collection of layers, blocks, and models re-implemented in tfjs.

Soon it will also contain benchmarks for preformsnce in JS.

Why?

I tried to impliment the famous decoder only GPT in tfjs and I noticed there was a lot missing in the standard lib that you would find in torch or tensorflow.

Not even attention was available.

I Decided why not start implimenting layers here. This will become my playground for implementing papers. 
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
  - [x] Stacked Shifted Window Blocks
- [x] models. (MobileVIT)
- [ ] Dilated Attention
- [ ] flash attention/Memory Aware Attention(Implement it naively from the torch repo)
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
