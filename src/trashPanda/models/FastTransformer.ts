// experimental custom model with mobilevit blocks interspersed with swin
//
// inspired by papers that interleave conv and transformers for speed gains. 
//
// this model interleaves conv based attention model mobilevit with real attention 
// effecient swin transformers and MobileNet2 conv blocks also seen in mobilevit
// 
// the swin blockd sre bissed towards the end sfter we have learned deep festures to take advantage
// of the more expensive large more accurate attention blocks.
//
// we trade off some of the speed of mobilevit for a fast but deeper architecture for scaling
// to web scale compute.
import * as tfjs from "@tfjs/core"
import { SwinBlock } from "./models/swinTransformer"
import { MobileViTBlock, MV2Block } from "./models/mobilevit"

class FastTransformer {
    // output could be class or image or upscale
    constructor(image_size, dims, channels, num_classes, expansion = 4, kernel_size = 3, patch_size = [2, 2], ouput="class") {
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
        
        this.decoder = new Decoder(output, channels, num_classes, num_feat, num_out_ch, upscale);
    }

    call(input) {
        let x = this.conv1.apply(input);

        x = this.transformer(x);
        x = this.conv2.apply(x);

        x = this.decoder.call(x);

        return x;
    }
}

class Decoder {
    constructor(output, channels, upscale) {
        this.output = output;
        this.upscale = upscale;

        if (this.output === 'upscale' || this.output === 'segmentation') {
            this.conv_before_upsample = tf.layers.conv2d({ filters: channels[channels.length - 2], kernelSize: 3, padding: 'same', activation: 'relu' });
            this.conv_up1 = tf.layers.conv2d({ filters: channels[channels.length - 2], kernelSize: 3, padding: 'same' });
            this.conv_up2 = tf.layers.conv2d({ filters: channels[channels.length - 2], kernelSize: 3, padding: 'same' });
            this.conv_hr = tf.layers.conv2d({ filters: channels[channels.length - 2], kernelSize: 3, padding: 'same' });
            this.conv_last = tf.layers.conv2d({ filters: 1, kernelSize: 3, padding: 'same', activation: 'sigmoid' }); // Binary output
            this.lrelu = tf.layers.leakyReLU({ alpha: 0.2 });
        } else {
            this.pool = tf.layers.averagePooling2d({ poolSize: [channels[channels.length - 2] // 32, channels[channels.length - 2] // 32], strides: 1 });
            this.fc = tf.layers.dense({ units: channels[channels.length - 1], useBias: false });
        }
    }

    call(x) {
        if (this.output === 'upscale') {
            x = this.conv_before_upsample.apply(x);
            x = this.lrelu.apply(this.conv_up1.apply(tf.image.resizeNearestNeighbor(x, [x.shape[1] * 2, x.shape[2] * 2])));
            if (this.upscale === 4) {
                x = this.lrelu.apply(this.conv_up2.apply(tf.image.resizeNearestNeighbor(x, [x.shape[1] * 2, x.shape[2] * 2])));
            }
            x = this.conv_last.apply(this.lrelu.apply(this.conv_hr.apply(x)));
        } else if (this.output === 'segmentation') {
            // Binary segmentation mask using similar logic to upscale
            x = this.conv_before_upsample.apply(x);
            x = this.lrelu.apply(this.conv_up1.apply(tf.image.resizeNearestNeighbor(x, [x.shape[1] * 2, x.shape[2] * 2])));
            x = this.conv_last.apply(this.lrelu.apply(this.conv_hr.apply(x))); // Binary mask with sigmoid activation
        } else {
            x = this.pool.apply(x);
            x = x.reshape([x.shape[0], -1]);
            x = this.fc.apply(x);
        }

        return x;
    }
}


function fastTransformerXXS() {
    const dims = [128, 128, 128];
    const dims = [128, 256, 512];
    // const channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320];
    const channels = [16, 16, 16, 16, 24, 24, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48];
    return new MobileViT([256, 256], dims, channels, 1000, 2);
}
