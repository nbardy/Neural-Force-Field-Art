# TrashPanda JS

<img width="300" src="out-0.png"></img>

Trash Panda JS is a library for writing modern architectures in tensorflow.js

It is a collection of layers, and blocks re-implemented in tfjs. Hopefully, Some of them will
be upstreamed to @tfjs/layers and other will remain research.

So far

```
import * as tp from "./trashPanda"

tp.linalg.normalize
tp.models.Transformers
tp.models.layers.MultiheadAttention
