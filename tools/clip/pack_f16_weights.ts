/**
 * Pack existing MobileCLIP f32 weight blobs into raw IEEE-fp16 scalar blobs.
 *
 * This is the first gate for the CLIP f16 fork: preserve the plan's logical
 * scalar indexing, halve the payload, and avoid doing conversion at page load.
 *
 *   bun tools/clip/pack_f16_weights.ts
 *   WEIGHTS=weights_train.bin bun tools/clip/pack_f16_weights.ts
 */
import { readFileSync, writeFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const requested = process.env.WEIGHTS;
const inputs = requested ? [requested] : ["weights.bin", "weights_train.bin"];

function f32ToF16Bits(value: number): number {
  if (Number.isNaN(value)) return 0x7e00;
  if (value === Infinity) return 0x7c00;
  if (value === -Infinity) return 0xfc00;

  const sign = value < 0 || Object.is(value, -0) ? 0x8000 : 0;
  const abs = Math.abs(value);
  if (abs === 0) return sign;

  if (abs >= 65504) return sign | 0x7bff;
  if (abs < 5.960464477539063e-8) return sign;

  if (abs < 0.00006103515625) {
    const mantissa = Math.round(abs / 5.960464477539063e-8);
    return sign | Math.max(1, Math.min(0x3ff, mantissa));
  }

  const exponent = Math.floor(Math.log2(abs));
  const halfExp = exponent + 15;
  const scaled = abs / 2 ** exponent;
  let mantissa = Math.round((scaled - 1) * 1024);
  let exp = halfExp;
  if (mantissa === 1024) {
    mantissa = 0;
    exp += 1;
  }
  if (exp >= 31) return sign | 0x7bff;
  return sign | (exp << 10) | (mantissa & 0x3ff);
}

function packOne(file: string): void {
  const srcPath = join(MODEL_DIR, file);
  const dstPath = join(MODEL_DIR, file.replace(/\.bin$/, "_f16.bin"));
  const buf = readFileSync(srcPath);
  if (buf.byteLength % 4 !== 0) {
    throw new Error(`${file}: byte length ${buf.byteLength} is not divisible by 4`);
  }
  const src = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const dst = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) dst[i] = f32ToF16Bits(src[i]);
  writeFileSync(dstPath, Buffer.from(dst.buffer, dst.byteOffset, dst.byteLength));

  const srcBytes = statSync(srcPath).size;
  const dstBytes = statSync(dstPath).size;
  const ratio = dstBytes / srcBytes;
  if (dst.length !== src.length) {
    throw new Error(`${file}: f16 scalars ${dst.length} != f32 scalars ${src.length}`);
  }
  console.log(
    `${file} -> ${file.replace(/\.bin$/, "_f16.bin")}: ` +
      `${src.length} scalars, ${(srcBytes / 1024 / 1024).toFixed(1)} MB -> ` +
      `${(dstBytes / 1024 / 1024).toFixed(1)} MB (${ratio.toFixed(3)}x)`
  );
}

for (const file of inputs) packOne(file);
