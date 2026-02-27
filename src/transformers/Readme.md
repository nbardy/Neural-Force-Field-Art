# Transformers.js

<img width="300" src="out-0.png"></img>

Transformers.js is a library for writing modern attention architectures in tensorflow.js

It is a collection of layers, blocks, and models re-implemented in tfjs. Transforming resesrch into code thst can run in the web.

### Why?

I tried to impliment the famous decoder only GPT in tfjs and I noticed there was a lot missing in the standard lib that you would find in torch or tensorflow.

Not even attention was available.

I Decided why not start implimenting layers here. This will become my playground for implementing papers.

### Whats Here

warning: code is written, but untested. will be getting it sll fully working slowly.

Demos:

- [x] Write a small web page that will train a GPT-tiny on an iphone.

Models and Layers from Research

- [x] MHSA
- [x] GPT
- [x] Transformer
  - [x] BPE
  - [x] Training scripts
  - [ ] Train Model      
- [x] Rotary Embeddings
- [ ] Local attention
- [ ] Dilated Global Residual (My idea for a simple speedup)
  - [x] Windowed Attention
- [x] SWIN Trasnformer
  - [x] Windowed Attention
  - [x] Shifted Windows
  - [x] Stacked Shifted Window Blocks
- [x] MobileVIT
- [ ] Dilated Attention
- [ ] Flash attention/Memory Aware Attention(Implement it naively from the torch repo)
  - [ ] Re-Implement with tfjs specific memory management (will need to benchmark with WEBGPU and WebGL)

### Experimental Code I came up with


Models:

- [x] FastTransformer(combines Swin and MobileViT)

Embeddings:

- [x] rotateEmbedding
- [x] sphericalRotationEmbedding
- [x] octoniomRotarionalEmbeddings
- [x] Max Embeddings(feature engineered Point Embeddings)

### Usage:

```
import * as tp from "./trashPanda"

tp.linalg.normalize
tp.models.Transformers
tp.models.layers.MultiheadAttention
```

### Contributions 

Contirbutions are welcome. I want to grow this collection to encompass all of the attention implimentstjons thst slready [exist in torch]()
