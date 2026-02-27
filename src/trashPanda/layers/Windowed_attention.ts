import * as tf from '@tensorflow/tfjs';

class PatchEmbed {
  img_size: [number, number];
  patch_size: [number, number];
  patches_resolution: [number, number];
  num_patches: number;
  in_chans: number;
  embed_dim: number;
  proj: tf.layers.Conv2D;
  norm?: tf.layers.Layer;

  constructor(img_size = 224, patch_size = 4, in_chans = 3, embed_dim = 96, norm_layer?: tf.layers.Layer) {
    this.img_size = Array.isArray(img_size) ? img_size : [img_size, img_size];
    this.patch_size = Array.isArray(patch_size) ? patch_size : [patch_size, patch_size];
    this.patches_resolution = [this.img_size[0] / this.patch_size[0], this.img_size[1] / this.patch_size[1]];
    this.num_patches = this.patches_resolution[0] * this.patches_resolution[1];

    this.in_chans = in_chans;
    this.embed_dim = embed_dim;

    this.proj = tf.layers.conv2d({
      filters: embed_dim,
      kernelSize: this.patch_size,
      strides: this.patch_size,
      inputShape: [this.img_size[0], this.img_size[1], in_chans]
    });

    if (norm_layer) {
      this.norm = norm_layer;
    }
  }

  forward(x: tf.Tensor) {
    const [B, H, W, C] = x.shape;

    if (H !== this.img_size[0] || W !== this.img_size[1]) {
      throw new Error(`Input image size (${H}*${W}) doesn't match model (${this.img_size[0]}*${this.img_size[1]}).`);
    }

    x = this.proj.apply(x) as tf.Tensor;  // BxHoxWoxC
    x = tf.reshape(x, [B, this.num_patches, this.embed_dim]); // Bx(Ph*Pw)xC

    if (this.norm) {
      x = this.norm.apply(x) as tf.Tensor;
    }

    return x;
  }

  flops() {
    const Ho = this.patches_resolution[0];
    const Wo = this.patches_resolution[1];
    let flops = Ho * Wo * this.embed_dim * this.in_chans * (this.patch_size[0] * this.patch_size[1]);
    if (this.norm) {
      flops += Ho * Wo * this.embed_dim;
    }
    return flops;
  }
}


function window_partition(x, window_size) {
    const [B, H, W, C] = x.shape;
    const xReshaped = tf.reshape(x, [B, H / window_size, window_size, W / window_size, window_size, C]);
    const windows = tf.reshape(tf.transpose(xReshaped, [0, 1, 3, 2, 4, 5]), [-1, window_size, window_size, C]);
    return windows;
}

function window_reverse(windows, window_size, H, W) {
  const B = windows.shape[0] / (H * W / window_size / window_size);
  const x = tf.reshape(windows, [B, H / window_size, W / window_size, window_size, window_size, -1]);
  const xReversed = tf.transpose(x, [0, 1, 3, 2, 4, 5]).reshape([B, H, W, -1]);
  return xReversed; // BxHxWxC
}

class WindowAttention {
    constructor(dim, window_size, num_heads, qkv_bias=true, qk_scale=null, attn_drop=0., proj_drop=0.) {
        this.dim = dim;
        this.window_size = window_size;
        this.num_heads = num_heads;
        const head_dim = dim / num_heads;
        this.scale = qk_scale || Math.pow(head_dim, -0.5);

        // Relative position bias table
        this.relative_position_bias_table = tf.variable(tf.zeros([(2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads]));

        // Relative position index
        const coords_h = tf.range(0, this.window_size[0]);
        const coords_w = tf.range(0, this.window_size[1]);
        const coords = tf.stack(tf.meshgrid(coords_h, coords_w));
        const coordsFlatten = tf.reshape(coords, [2, -1]);
        let relativeCoords = coordsFlatten.expandDims(2).sub(coordsFlatten.expandDims(1));
        relativeCoords = relativeCoords.add(this.window_size[0] - 1).mul(2 * this.window_size[1] - 1);
        this.relative_position_index = relativeCoords.sum(-1).reshape(this.window_size[0] * this.window_size[1], this.window_size[0] * this.window_size[1]);

        this.qkv = tf.layers.dense({ units: dim * 3, useBias: qkv_bias });
        this.attn_drop = tf.layers.dropout({ rate: attn_drop });
        this.proj = tf.layers.dense({ units: dim });
        this.proj_drop = tf.layers.dropout({ rate: proj_drop });
        this.softmax = tf.layers.softmax();
    }

    async forward(x, mask=null) {
        const B_ = x.shape[0];
        const N = x.shape[1];
        const C = x.shape[2];
        const qkv = this.qkv.apply(x).reshape([B_, N, 3, this.num_heads, C / this.num_heads]).transpose([2, 0, 3, 1, 4]);
        const q = qkv.slice([0, 0, 0, 0, 0], [1, -1, -1, -1, -1]).squeeze();
        const k = qkv.slice([1, 0, 0, 0, 0], [1, -1, -1, -1, -1]).squeeze();
        const v = qkv.slice([2, 0, 0, 0, 0], [1, -1, -1, -1, -1]).squeeze();
        let attn = q.mul(this.scale).matMul(k.transpose(-2, -1));

        const relative_position_bias = tf.gather(this.relative_position_bias_table, this.relative_position_index.flatten()).reshape(this.window_size[0] * this.window_size[1], this.window_size[0] * this.window_size[1], -1).transpose([2, 0, 1]);
        attn = attn.add(relative_position_bias.expandDims(0));

        if (mask !== null) {
            const nW = mask.shape[0];
            attn = attn.reshape(B_ / nW, nW, this.num_heads, N, N).add(mask.expandDims(1).expandDims(0)).reshape(-1, this.num_heads, N, N);
            attn = this.softmax.apply(attn);
        } else {
            attn = this.softmax.apply(attn);
        }

        attn = this.attn_drop.apply(attn);

        x = attn.matMul(v).transpose([1, 2, 0, 3]).reshape([B_, N, C]);
        x = this.proj.apply(x);
        x = this.proj_drop.apply(x);
        return x;
    }
}

class SwinTransformerBlock extends tf.layers.Layer {
  constructor({
  dim,
  inputResolution,
  numHeads,
  windowSize = 7,
  shiftSize = 0,
  mlpRatio = 4.0,
  qkvBias = true,
  qkScale = null,
  drop = 0.0,
  attnDrop = 0.0,
  dropPath = 0.0,
  actLayer = tf.activations.gelu,
  normLayer = tf.layers.LayerNormalization,
  fusedWindowProcess = false,
}) {
  super({});
  this.dim = dim;
  this.inputResolution = inputResolution;
  this.numHeads = numHeads;
  this.windowSize = windowSize;
  this.shiftSize = shiftSize;
  this.mlpRatio = mlpRatio;

  if (Math.min(...this.inputResolution) <= this.windowSize) {
    this.shiftSize = 0;
    this.windowSize = Math.min(...this.inputResolution);
  }

  this.norm1 = normLayer({ axis: -1 });
  this.attn = new WindowAttention({
    dim,
    windowSize,
    numHeads,
    qkvBias,
    qkScale,
    attnDrop,
    projDrop: drop,
  });

  this.dropPath = dropPath > 0.0 ? new DropPath(dropPath) : tf.layers.Lambda(x => x);
  this.norm2 = normLayer({ axis: -1 });
  const mlpHiddenDim = dim * mlpRatio;
  this.mlp = new Mlp(dim, mlpHiddenDim, actLayer, drop);
  this.fusedWindowProcess = fusedWindowProcess;

  if (this.shiftSize > 0) {
    const [H, W] = this.inputResolution;
    const imgMask = tf.zeros([1, H, W, 1]);
    const hSlices = [
      tf.slice(imgMask, [0, 0, 0, 0], [-1, H - this.windowSize, -1, -1]),
      tf.slice(imgMask, [0, H - this.windowSize, 0, 0], [-1, this.windowSize - this.shiftSize, -1, -1]),
      tf.slice(imgMask, [0, H - this.shiftSize, 0, 0], [-1, this.shiftSize, -1, -1])
    ];
    const wSlices = [
      tf.slice(imgMask, [0, 0, 0, 0], [-1, -1, W - this.windowSize, -1]),
      tf.slice(imgMask, [0, 0, W - this.windowSize, 0], [-1, -1, this.windowSize - this.shiftSize, -1]),
      tf.slice(imgMask, [0, 0, W - this.shiftSize, 0], [-1, -1, this.shiftSize, -1])
    ];
    let cnt = 0;
    for (const h of hSlices) {
      for (const w of wSlices) {
        imgMask.assign(tf.add(imgMask, cnt, h, w));
        cnt += 1;
      }
    }

    const maskWindows = windowPartition(imgMask, this.windowSize);
    const maskWindowsReshaped = tf.reshape(maskWindows, [-1, this.windowSize * this.windowSize]);
    const attnMask = tf.sub(tf.expandDims(maskWindowsReshaped, 1), tf.expandDims(maskWindowsReshaped, 2));
    this.attnMask = tf.where(attnMask.notEqual(0), tf.fill(attnMask.shape, -100.0), tf.fill(attnMask.shape, 0.0));
  } else {
    this.attnMask = null;
  }
}


  call(inputs) {
    const [H, W] = this.inputResolution;
    const [B, L, C] = inputs.shape;
    const shortcut = inputs;
    let x = this.norm1.apply(inputs);
    x = tf.reshape(x, [B, H, W, C]);

    if (this.shiftSize > 0) {
      const shiftedX = tf.roll(x, [-this.shiftSize, -this.shiftSize], [1, 2]);
      const xWindows = window_partition(shiftedX, this.windowSize);
      const xWindowsReshaped = tf.reshape(xWindows, [-1, this.windowSize * this.windowSize, C]);

      const attnWindows = this.attn.call(xWindowsReshaped, this.attnMask);
      const attnWindowsReshaped = tf.reshape(attnWindows, [-1, this.windowSize, this.windowSize, C]);

      let shiftedXReverse;
      if (this.fusedWindowProcess) {
        shiftedXReverse = WindowProcessReverse.apply(attnWindowsReshaped, B, H, W, C, this.shiftSize, this.windowSize);
      } else {
        shiftedXReverse = window_reverse(attnWindowsReshaped, this.windowSize, H, W);
        x = tf.roll(shiftedXReverse, [this.shiftSize, this.shiftSize], [1, 2]);
      }
    } else {
      const xWindows = window_partition(x, this.windowSize);
      const xWindowsReshaped = tf.reshape(xWindows, [-1, this.windowSize * this.windowSize, C]);

      const attnWindows = this.attn.call(xWindowsReshaped, this.attnMask);
      const attnWindowsReshaped = tf.reshape(attnWindows, [-1, this.windowSize, this.windowSize, C]);
      x = window_reverse(attnWindowsReshaped, this.windowSize, H, W);
    }

    x = tf.reshape(x, [B, H * W, C]);
    x = x.add(this.dropPath.apply(shortcut));
    x = x.add(this.dropPath.apply(this.mlp.apply(this.norm2.apply(x))));

    return x;
  }
}

class PatchMerging extends tf.layers.Layer {
  inputResolution: [number, number];
  dim: number;
  reduction: tf.layers.Dense;
  norm: tf.layers.LayerNormalization;

  constructor(inputResolution: [number, number], dim: number, normLayer = tf.layers.LayerNormalization) {
    super({});
    this.inputResolution = inputResolution;
    this.dim = dim;
    this.reduction = tf.layers.dense({ units: 2 * dim, inputShape: [4 * dim], useBias: false });
    this.norm = normLayer({ axis: -1, epsilon: 1e-6 });
  }

  call(x: tf.Tensor) {
    const [H, W] = this.inputResolution;
    const [B, L, C] = x.shape as [number, number, number];
    if (L !== H * W || H % 2 !== 0 || W % 2 !== 0) {
      throw new Error("input feature has wrong size");
    }

    const xReshape = tf.reshape(x, [B, H, W, C]);
    const x0 = xReshape.slice([0, 0, 0, 0], [B, H / 2, W / 2, C]);
    const x1 = xReshape.slice([0, 1, 0, 0], [B, H / 2, W / 2, C]);
    const x2 = xReshape.slice([0, 0, 1, 0], [B, H / 2, W / 2, C]);
    const x3 = xReshape.slice([0, 1, 1, 0], [B, H / 2, W / 2, C]);

    const xConcat = tf.concat([x0, x1, x2, x3], -1);
    const xView = tf.reshape(xConcat, [B, -1, 4 * C]);

    const xNorm = this.norm.apply(xView) as tf.Tensor;
    return this.reduction.apply(xNorm) as tf.Tensor;
  }
}

class SwinTransformer {
  numClasses: number;
  numLayers: number;
  embedDim: number;
  ape: boolean;
  patchNorm: boolean;
  numFeatures: number;
  mlpRatio: number;
  patchEmbed: PatchEmbed; // Define PatchEmbed class accordingly
  absolutePosEmbed?: tf.Tensor;
  posDrop: tf.layers.Dropout;
  layers: BasicLayer[]; // Define BasicLayer class accordingly
  norm: tf.layers.LayerNormalization;
  avgpool: tf.layers.GlobalAveragePooling1D;
  head: tf.layers.Dense;

  constructor(img_size = 224, patch_size = 4, in_chans = 3, num_classes = 1000,
              embed_dim = 96, depths = [2, 2, 6, 2], num_heads = [3, 6, 12, 24],
              window_size = 7, mlp_ratio = 4., qkv_bias = true, qk_scale = null,
              drop_rate = 0., attn_drop_rate = 0., drop_path_rate = 0.1,
              norm_layer = tf.layers.LayerNormalization, ape = false, patch_norm = true,
              use_checkpoint = false, fused_window_process = false) {

    this.numClasses = num_classes;
    this.numLayers = depths.length;
    this.embedDim = embed_dim;
    this.ape = ape;
    this.patchNorm = patch_norm;
    this.numFeatures = embed_dim * Math.pow(2, this.numLayers - 1);
    this.mlpRatio = mlp_ratio;

    // split image into non-overlapping patches
    this.patchEmbed = new PatchEmbed(img_size, patch_size, in_chans, embed_dim, this.patchNorm ? norm_layer : null);
    const numPatches = this.patchEmbed.numPatches;
    const patchesResolution = this.patchEmbed.patchesResolution;

    // absolute position embedding
    if (this.ape) {
      this.absolutePosEmbed = tf.zeros([1, numPatches, embed_dim]);
    }

    this.posDrop = tf.layers.dropout({ rate: drop_rate });

    // stochastic depth
    const dpr = tf.linspace(0, drop_path_rate, depths.reduce((a, b) => a + b)).arraySync(); // stochastic depth decay rule

    // build layers
    this.layers = [];
    for (let iLayer = 0; iLayer < this.numLayers; iLayer++) {
      const layer = new BasicLayer(this.embedDim * Math.pow(2, iLayer),
                                   [patchesResolution[0] / Math.pow(2, iLayer), patchesResolution[1] / Math.pow(2, iLayer)],
                                   depths[iLayer], num_heads[iLayer], window_size, this.mlpRatio,
                                   qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                                   dpr.slice(depths.slice(0, iLayer).reduce((a, b) => a + b),
                                             depths.slice(0, iLayer + 1).reduce((a, b) => a + b)),
                                   norm_layer, iLayer < this.numLayers - 1 ? PatchMerging : null, // Define PatchMerging if needed
                                   use_checkpoint, fused_window_process);
      this.layers.push(layer);
    }

    this.norm = norm_layer({ epsilon: 1e-5 });
    this.avgpool = tf.layers.globalAveragePooling1d();
    this.head = num_classes > 0 ? tf.layers.dense({ units: num_classes }) : null; // Handle identity operation if needed
  }

  forwardFeatures(x: tf.Tensor) {
    x = this.patchEmbed.call(x);
    if (this.ape) {
      x = x.add(this.absolutePosEmbed);
    }
    x = this.posDrop.apply(x) as tf.Tensor;

    for (const layer of this.layers) {
      x = layer.call(x);
    }

    x = this.norm.apply(x) as tf.Tensor;
    x = this.avgpool.apply(x.transpose([0, 2, 1])) as tf.Tensor; // B C 1
    x = x.flatten(1);
    return x;
  }

  forward(x: tf.Tensor) {
    x = this.forwardFeatures(x);
    if (this.head) {
      x = this.head.apply(x) as tf.Tensor;
    }
    return x;
  }
}

