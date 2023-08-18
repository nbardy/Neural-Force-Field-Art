// experimental custom model with mobilevit blocks interspersed with swin
//
// inspired by papers that interleave conv and transformers for speed gains. 
//
// transforming mobilvit into a much deep versio, slows down mobilecit, but allows us to scale
// compite with depth
// todo: replace swin blocks with a faster windowed attention with global residual
// first train with swin to get baseline.
import * as tfjs from "@tfjs/core"
import {SwinBlock} from "./swinTransformer.ts"

class FastTransformer {
    constructor(image_size, dims, channels, num_classes, expansion = 4, kernel_size = 3, patch_size = [2, 2]) {
        const [ih, iw] = image_size;
        const [ph, pw] = patch_size;
        if (ih % ph !== 0 || iw % pw !== 0) {
            throw new Error("Image size should be divisible by patch size");
        }

        const L = [4, 4, 4];
        const swinL = [4, 2, 2];
   
        this.conv1 = conv_nxn_bn(3, channels[0], 2);

        this.transformer = tfjs.seq([
            new MV2Block(channels[0], channels[1], 1, expansion),
            new MV2Block(channels[1], channels[2], 1, expansion),         
            new MV2Block(channels[2], channels[3], 2, expansion),
            new MV2Block(channels[2], channels[3], 1, expansion),
            new MV2Block(channels[3], channels[4], 1, expansion),
            new MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, Math.floor(dims[0] * 2)),
            new MV2Block(channels[5], channels[6], 1, expansion),
            new MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, Math.floor(dims[1] * 2)),
            new MV2Block(channels[7], channels[8], 1, expansion),
            new SwinBlock(swin_dims[0], swin_L[0], channels[9]),
            new MV2Block(channels[9], channels[10], 1, expansion),
            new MobileViTBlock(dims[2], L[2], channels[11], kernel_size, patch_size, Math.floor(dims[2] * 2)),
            new SwinBlock(swin_dims[1], swin_L[1], channels[12]),
            new MV2Block(channels[12], channels[13], 1, expansion),
            new MV2Block(channels[13], channels[14], 1, expansion),
            new SwinBlock(swin_dims[2], swin_L[2], channels[15]),
            new MV2Block(channels[15], channels[16], 1, expansion),
        ])

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

function fastTransformerXXS() {
    const dims = [128, 128, 128];
    const dims = [64, 128, 128];
    // const channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320];
    const channels = [16, 16, 16, 16, 24, 24, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48];
    return new MobileViT([256, 256], dims, channels, 1000, 2);
}
