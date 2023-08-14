export const rotate2d = (vec, theta) => {
  const [x, y] = vec;
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);

  return [x * cos - y * sin, x * sin + y * cos];
};

// rotate hue
//
// rotate across L axis to rotate A,B around hue plane
export const rotateAB = (rgb, theta) => {
  const lab = rgb2lab(rgb);
  const [l, A, B] = lab;

  const axis = [A, B];
  const rotated = rotate2d(axis, theta);

  return lab2rgb([l, rotated[0], rotated[1]]);
};

const skewAll = (rgb, xyz) => {
  const lab = rgb2lab(rgb);

  // zip and map pow
  return lab2rgb(lab.map((v, i) => Math.pow(v, xyz[i])));
};

/*
 * LAB
 */

export function rgb2lab(rgb) {
  let r = rgb[0] / 255;
  let g = rgb[1] / 255;
  let b = rgb[2] / 255;

  // Convert to XYZ color space
  let x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047;
  let y = (r * 0.2126729 + g * 0.7151522 + b * 0.072175) / 1.0;
  let z = (r * 0.0193339 + g * 0.119192 + b * 0.9503041) / 1.08883;

  // Apply the D65 illuminant
  [x, y, z] = [x, y, z].map((v) =>
    v > 0.008856 ? Math.pow(v, 1 / 3) : v * 7.787 + 16 / 116
  );

  let L = 116 * y - 16;
  let A = (x - y) * 500;
  let B = (y - z) * 200;

  return [L, A, B];
}

export function lab2rgb(lab) {
  let [L, A, B] = lab;
  let y = (L + 16) / 116;
  let x = A / 500 + y;
  let z = y - B / 200;

  // D65 illuminant
  [x, y, z] = [x, y, z].map((v) =>
    Math.pow(v, 3) > 0.008856 ? Math.pow(v, 3) : (v - 16 / 116) / 7.787
  );
  x *= 0.95047;
  y *= 1.0;
  z *= 1.08883;

  let r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314;
  let g = x * -0.969266 + y * 1.8760108 + z * 0.041556;
  let b = x * 0.0556434 + y * -0.2040259 + z * 1.0572252;

  [r, g, b] = [r, g, b].map((v) =>
    v > 0.0031308 ? 1.055 * Math.pow(v, 1 / 2.4) - 0.055 : 12.92 * v
  );

  return [r * 255, g * 255, b * 255];
}

// Shader function for LAB conversion
export const labConversionFunctions = `
  vec3 rgb2lab(vec3 rgb) {
    // D65 illuminant
    const mat3 rgb2xyz = mat3(
      0.4124564, 0.3575761, 0.1804375,
      0.2126729, 0.7151522, 0.0721750,
      0.0193339, 0.1191920, 0.9503041
    );
    const vec3 rgb2xyz_offset = vec3(0.95047, 1.0, 1.08883);

    // Convert to XYZ color space
    vec3 xyz = rgb2xyz * rgb;
    xyz = pow(xyz, vec3(1.0 / 2.4));
    xyz = (xyz > 0.008856) ? pow(xyz, vec3(1.0 / 3.0)) : (7.787 * xyz) + (16.0 / 116.0);
    xyz *= rgb2xyz_offset;

    // Convert to LAB color space
    vec3 lab;
    lab.x = (116.0 * xyz.y) - 16.0;
    lab.y = 500.0 * (xyz.x - xyz.y);
    lab.z = 200.0 * (xyz.y - xyz.z);
    lab.y = (lab.y > 127.0) ? lab.y - 255.0 : lab.y;
    lab.z = (lab.z > 127.0) ? lab.z - 255.0 : lab.z;
    return lab;

  }

  vec3 lab2rgb(vec3 lab) {
    // D65 illuminant
    const mat3 xyz2rgb = mat3(
        3.2404542, -1.5371385, -0.4985314,
        -0.9692660, 1.8760108, 0.0415560,
        0.0556434, -0.2040259, 1.0572252
    );

    // Convert to XYZ color space
    vec3 xyz;
    xyz.y = (lab.x + 16.0) / 116.0;
    xyz.x = lab.y / 500.0 + xyz.y;
    xyz.z = xyz.y - lab.z / 200.0;
    xyz = pow(xyz, vec3(3.0));
    xyz = (xyz > 0.008856) ? xyz : (xyz - (16.0 / 116.0)) / 7.787;

    // Convert to RGB color space
    vec3 rgb = xyz2rgb * xyz;
    rgb = pow(rgb, vec3(2.4));
    return rgb;
  }
`;

import * as tf from "@tensorflow/tfjs";

export const rgb2labTensor = (rgb) => {
  // Convert to XYZ color space
  const rgbTensor = tf.tensor1d(rgb).div(tf.scalar(255));
  const rgb2xyz = tf.tensor2d(
    [
      [0.4124564, 0.3575761, 0.1804375],
      [0.2126729, 0.7151522, 0.072175],
      [0.0193339, 0.119192, 0.9503041],
    ],
    [3, 3]
  );

  const xyzOffset = tf.tensor1d([0.95047, 1.0, 1.08883]);

  let xyz = rgbTensor.dot(rgb2xyz);
  xyz = xyz.pow(tf.scalar(1 / 2.4));
  const greaterThanThreshold = xyz.greater(tf.scalar(0.008856));
  xyz = greaterThanThreshold.where(
    xyz.pow(tf.scalar(1 / 3)),
    xyz.mul(tf.scalar(7.787)).add(tf.scalar(16 / 116))
  );
  xyz = xyz.mul(xyzOffset);

  // Convert to LAB color space
  const lab = tf.tidy(() => {
    const lab = tf
      .tensor1d([116, 500, 200])
      .mul(
        tf
          .tensor1d([1, -1, -1])
          .mul(xyz.slice([1], [1]).sub(xyz.slice([0], [1])))
      );

    return tf.concat(
      [
        lab.slice([0], [1]).sub(tf.scalar(16)),
        lab.slice([1], [1]),
        lab.slice([2], [1]),
      ],
      0
    );
  });

  return lab;
};

export const lab2rgbTensor = (lab) => {
  // D 65 illuminant
  const xyz2rgb = tf.tensor2d(
    [
      [3.2404542, -1.5371385, -0.4985314],
      [-0.969266, 1.8760108, 0.041556],
      [0.0556434, -0.2040259, 1.0572252],
    ],
    [3, 3]
  );

  // Steps:
  // 1. Convert to XYZ color space
  // 2. Convert to RGB color space
  // 3. Convert to 0-255 range

  // Convert to XYZ color space
  const xyzOffset = tf.tensor1d([0.95047, 1.0, 1.08883]);
  let xyz = tf.tidy(() => {
    const labOffset = tf.tensor1d([16, 500, 200]);
    const lab2xyz = tf.tensor2d(
      [
        [1 / 116, 1 / 116, 1 / 116],
        [1 / 500, 0, 0],
        [0, 0, -1 / 200],
      ],
      [3, 3]
    );

    let labTensor = lab.add(labOffset);
    let v = labTensor.dot(lab2xyz);
    v = v.pow(tf.scalar(3));
    const greaterThanThreshold = v.greater(tf.scalar(0.008856));
    v = greaterThanThreshold.where(
      v,
      v.sub(tf.scalar(16 / 116)).div(tf.scalar(7.787))
    );

    return v.mul(xyzOffset);
  });

  // Convert to RGB color space
  let rgb = xyz.dot(xyz2rgb);
  rgb = rgb.pow(tf.scalar(2.4));

  // Convert to 0-255 range
  rgb = rgb.mul(tf.scalar(255));
  rgb = rgb.clipByValue(tf.scalar(0), tf.scalar(255));

  return rgb;
};

// bump power(amplify, keep in 0-1)
export const skewL = (rgb, a) => {
  const lab = rgb2lab(rgb);
  const [l, A, B] = lab;

  return lab2rgb([Math.pow(l, a), A, B]);
};
