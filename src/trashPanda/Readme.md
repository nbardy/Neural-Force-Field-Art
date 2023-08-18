# TrashPanda JS

<img width="300" src="out-0.png"></img>

Trash Panda JS is a library for writing modern attention architectures in tensorflow.js

It is a collection of layers, blocks, and models re-implemented in tfjs.

Soon it will also contain benchmarks for preformsnce in JS.

### Why?

I tried to impliment the famous decoder only GPT in tfjs and I noticed there was a lot missing in the standard lib that you would find in torch or tensorflow.

Not even attention was available.

I Decided why not start implimenting layers here. This will become my playground for implementing papers.

### Whats Here

Demos:

- [ ] Write a small web page that will train a GPT-tiny on an iphone.

Models and Layers from Research

- [x] MHSA
- [x] GPT
- [x] Transformer
  - [x] BPE
  - [ ] Training scripts        
- [x] Rotary Embeddings
- [ ] Local attention
- [ ] Dilated Global Residual (My idea for a simple speedup)
  - [ ] Windowed Attention
- [x] Shfited Trasnformer
  - [x] Windowed Attention
  - [x] Shifted Windows
  - [x] Stacked Shifted Window Blocks
- [x] MobileVIT
- [ ] Dilated Attention
- [ ] Flash attention/Memory Aware Attention(Implement it naively from the torch repo)
  - [ ] Re-Implement with tfjs specific memory management (will need to benchmark with WEBGPU and WebGL)

### Experimental Code I came up with

Embeddings:
- [x] rotateEmbedding
- [x] sphericalRotationEmbedding
- [x] octoniomRotarionalEmbeddings
- [x] Max Embeddings(feature engineered Point Embeddings)

Usage
```
import * as tp from "./trashPanda"

tp.linalg.normalize
tp.models.Transformers
tp.models.layers.MultiheadAttention
```
