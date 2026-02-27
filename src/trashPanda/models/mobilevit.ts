import * as tf from '@tensorflow/tfjs';

function conv1x1BN(inp, oup) {
    return tf.layers.sequential({
        layers: [
            tf.layers.conv2d({filters: oup, kernelSize: 1, strides: 1, useBias: false, inputShape: [inp, inp, inp]}),
            tf.layers.layerNormalization(),
            tf.layers.activation({activation: 'swish'})
        ]
    });
}

function convNxNBN(inp, oup, kernelSize = 3, stride = 1) {
    return tf.layers.sequential({
        layers: [
            tf.layers.conv2d({filters: oup, kernelSize: kernelSize, strides: stride, useBias: false, padding: 'same', inputShape: [inp, inp, inp]}),
            tf.layers.layerNormalization(),
            tf.layers.activation({activation: 'swish'})
        ]
    });
}

// PreNorm Layer
class PreNorm {
    constructor(dim, fn) {
        this.norm = tf.layers.layerNormalization({axis: -1});
        this.fn = fn;
    }
    
    call(x, kwargs) {
        return this.fn(this.norm.apply(x), kwargs);
    }
}

// FeedForward Layer
class FeedForward {
    constructor(dim, hiddenDim, dropout = 0.0) {
        this.net = tf.layers.sequential({
            layers: [
                tf.layers.dense({units: hiddenDim, activation: 'swish'}),
                tf.layers.dropout({rate: dropout}),
                tf.layers.dense({units: dim}),
                tf.layers.dropout({rate: dropout})
            ]
        });
    }

    call(x) {
        return this.net.apply(x);
    }
}

// Attention Layer
class Attention {
    constructor(dim, heads = 8, dimHead = 64, dropout = 0.0) {
        this.innerDim = dimHead * heads;
        this.projectOut = !(heads === 1 && dimHead === dim);

        this.heads = heads;
        this.scale = Math.pow(dimHead, -0.5);

        this.attend = tf.layers.activation({activation: 'softmax'});
        this.toQKV = tf.layers.dense({units: this.innerDim * 3, useBias: false});

        this.toOut = this.projectOut ? tf.layers.sequential({
            layers: [
                tf.layers.dense({units: this.innerDim}),
                tf.layers.dropout({rate: dropout})
            ]
        }) : new tf.layers.Identity();
    }

    call(x) {
        // Handle q, k, v
        const qkv = this.toQKV.apply(x).split(3, -1);
        const [q, k, v] = qkv.map(t => tf.reshape(t, [t.shape[0], t.shape[1], this.heads, -1]));
        const dots = tf.matMul(q, k.transpose([-1, -2])).mul(this.scale);
        const attn = this.attend.apply(dots);
        const out = tf.matMul(attn, v);
        const outRearrange = tf.reshape(out, [out.shape[0], out.shape[1], -1]);
        return this.toOut.apply(outRearrange);
    }
}

class Transformer {
    constructor(dim, depth, heads, dimHead, mlpDim, dropout = 0.0) {
        this.layers = [];
        for (let _ = 0; _ < depth; _++) {
            this.layers.push([
                new PreNorm(dim, new Attention(dim, heads, dimHead, dropout)),
                new PreNorm(dim, new FeedForward(dim, mlpDim, dropout))
            ]);
        }
    }
    
    call(x) {
        for (let i = 0; i < this.layers.length; i++) {
            const [attn, ff] = this.layers[i];
            x = attn.call(x).add(x);
            x = ff.call(x).add(x);
        }
        return x;
    }
}


// MV2Block Layer
class MV2Block {
    constructor(inp, oup, stride = 1, expansion = 4, normType = "layer") {
        this.stride = stride;
        const norm = normType === "layer" ? tf.layers.layerNormalization : tf.layers.batchNormalization;
        assert(stride === 1 || stride === 2);

        let hiddenDim = Math.round(inp * expansion);
        this.useResConnect = this.stride === 1 && inp === oup;

        if (expansion === 1) {
            this.conv = tf.sequential([
                tf.layers.conv2d({filters: hiddenDim, kernelSize: 3, strides: stride, padding: 'same', useBias: false}),
                norm(),
                tf.layers.activation({activation: 'relu'}),
                tf.layers.conv2d({filters: oup, kernelSize: 1, strides: 1, useBias: false}),
                norm()
            ]);
        } else {
            this.conv = tf.sequential([
                tf.layers.conv2d({filters: hiddenDim, kernelSize: 1, strides: 1, useBias: false}),
                norm(),
                tf.layers.activation({activation: 'relu'}),
                tf.layers.conv2d({filters: hiddenDim, kernelSize: 3, strides: stride, padding: 'same', useBias: false}),
                norm(),
                tf.layers.activation({activation: 'relu'}),
                tf.layers.conv2d({filters: oup, kernelSize: 1, strides: 1, useBias: false}),
                norm()
            ]);
        }
    }

    call(x) {
        if (this.useResConnect) {
            return x.add(this.conv.apply(x));
        } else {
            return this.conv.apply(x);
        }
    }
}

// MobileViTBlock Layer
class MobileViTBlock {
    constructor(dim, depth, channel, kernelSize, patchSize, mlpDim, dropout = 0.0) {
        this.ph = patchSize[0];
        this.pw = patchSize[1];

        this.conv1 = conv_nxn_bn(channel, channel, kernelSize);
        this.conv2 = conv_1x1_bn(channel, dim);

        this.transformer = new Transformer(dim, depth, 4, 8, mlpDim, dropout);

        this.conv3 = conv_1x1_bn(dim, channel);
        this.conv4 = conv_nxn_bn(2 * channel, channel, kernelSize);
    }

    call(x) {
        let y = x.clone();

        // Local representations
        x = this.conv1.apply(x);
        x = this.conv2.apply(x);
        
        // Global representations
        let [b, h, w, d] = x.shape;
        x = tf.reshape(x, [b, this.ph, this.pw, h / this.ph, w / this.pw, d]);
        x = this.transformer.call(x);
        x = tf.reshape(x, [b, d, h, w]);

        // Fusion
        x = this.conv3.apply(x);
        x = tf.concat([x, y], -1);
        x = this.conv4.apply(x);
        return x;
    }
}


// MobileViT Model
class MobileViT {
    constructor(image_size, dims, channels, num_classes, expansion = 4, kernel_size = 3, patch_size = [2, 2]) {
        const [ih, iw] = image_size;
        const [ph, pw] = patch_size;
        if (ih % ph !== 0 || iw % pw !== 0) {
            throw new Error("Image size should be divisible by patch size");
        }

        const L = [2, 4, 3];

        this.conv1 = conv_nxn_bn(3, channels[0], 2);

        this.mv2 = [
            new MV2Block(channels[0], channels[1], 1, expansion),
            new MV2Block(channels[1], channels[2], 2, expansion),
            new MV2Block(channels[2], channels[3], 1, expansion),
            new MV2Block(channels[2], channels[3], 1, expansion), // Repeat
            new MV2Block(channels[3], channels[4], 2, expansion),
            new MV2Block(channels[5], channels[6], 2, expansion),
            new MV2Block(channels[7], channels[8], 2, expansion),
        ];
        
        this.mvit = [
            new MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, Math.floor(dims[0] * 2)),
            new MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, Math.floor(dims[1] * 4)),
            new MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, Math.floor(dims[2] * 4)),
        ];

        this.conv2 = conv_1x1_bn(channels[channels.length - 2], channels[channels.length - 1]);
        this.pool = tf.layers.averagePooling2d({ poolSize: [ih // 32, ih // 32], strides: 1 });
        this.fc = tf.layers.dense({ units: num_classes, useBias: false });
    }

    call(input) {
        let x = this.conv1.apply(input);

        x = this.mv2[0].call(x);

        x = this.mv2[1].call(x);
        x = this.mv2[2].call(x);
        x = this.mv2[3].call(x); // Repeat

        x = this.mv2[4].call(x);
        x = this.mvit[0].call(x);

        x = this.mv2[5].call(x);
        x = this.mvit[1].call(x);

        x = this.mv2[6].call(x);
        x = this.mvit[2].call(x);
        x = this.conv2.apply(x);

        x = this.pool.apply(x);
        x = x.reshape([x.shape[0], -1]); 
        x = this.fc.apply(x);

        return x;
    }
}


function mobilevitXXS() {
    const dims = [64, 80, 96];
    const channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320];
    return new MobileViT([256, 256], dims, channels, 1000, 2);
}

function mobilevitXS() {
    const dims = [96, 120, 144];
    const channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384];
    return new MobileViT([256, 256], dims, channels, 1000);
}

function mobilevitS() {
    const dims = [144, 192, 240];
    const channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640];
    return new MobileViT([256, 256], dims, channels, 1000);
}
