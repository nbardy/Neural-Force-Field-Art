// Particle:
//  [[x,y],  # Position
//   [x, y], # Velocity }
//
//
import * as tf from "@tensorflow/tfjs";
import { getEnv } from "../utils/env";

export function newField(config) {
  const w = config.width * config.density;
  const h = config.height * config.density;

  const size = Math.round(w * h);

  return tf.tidy(() => {
    console.log(w, h);
    const dir = tf.randomUniform([size], 0, 2 * Math.PI);

    // Create this as a force vector field
    const mag = tf.randomNormal(
      [size],
      config.initForceMagnitude,
      config.initForceStdDev,
      "float32",
      config.randomSeed
    );

    // I don't remember why I cast this away from a force vector.
    return dir.cos().mul(mag).stack(dir.sin().mul(mag), 1);
  });
}

export function newParticles(config) {
  return tf.tidy(() => {
    const posx = tf.randomUniform(
      [config.particleCount],
      0,
      config.width,
      "float32"
    );
    const posy = tf.randomUniform(
      [config.particleCount],
      0,
      config.height,
      "float32"
    );
    const pos = posx.stack(posy, 1);
    const vel = tf.zerosLike(pos);

    return [pos, vel];
  });
}

export function clipField(field, mag) {
  return tf.tidy(() => {
    return field.clipByValue(-mag, mag);
  });
}
// used for dev
// lots of these are not used
const defaultConfig = {
  width: window.innerWidth,
  height: window.innerHeight,
  resetRate: 0.000001,
  forceMagnitude: 0.8,
  friction: 0.911,
  maximumVelocity: 2.2,
  maximumForce: 20.2,
  particleCount: 1000,
  entropyDecay: 0.99,
  epochs: 3,
  drawField: false,
  backgroundColor: [12, 0, 34],
};

// config type
// partial
type Config = typeof defaultConfig;

export function updateParticles2(
  [pos, vel],
  model,
  configOverrides: Partial<Config> = {}
) {
  // merge with default config
  const config: Config = Object.assign({}, defaultConfig, configOverrides);

  return tf.tidy(() => {
    // Scale down to fit force field dimensions
    const particles = [pos, vel];

    const posNormalized = pos.div(
      tf.tensor2d([config.width, config.height], [1, 2])
    );

    // const posShifted = posNormalized.sub(tf.scalar(0.5));
    // pt("pshift", posShifted);
    // pt("p", posNormalized);
    const forces = model.predict(posNormalized);
    // pt("f", forces);
    //   tf.concat([posNormalized, velNormalized], axis)
    // );

    // Shift forces from 0,1 to -0.5,0.5
    const forcesShifted = forces.sub(tf.scalar(0.5));

    // Forces applied with relevant magnitude
    const forcesScaled = forcesShifted.mul(tf.scalar(config.forceMagnitude));

    pt("forces", forces);
    pt("fsshift", forcesShifted);
    pt("fscale", forcesScaled);

    const updatedVel = vel.add(forcesScaled).mul(tf.scalar(config.friction));
    const updatedPos = pos.add(updatedVel);

    // Wrap Positions
    // TODO: Make option to change this behavior(invert, reset, dissapear)

    // Slice to take only the X values then mod them by config.width to wrap around
    const posX = updatedPos.slice([0, 0], [-1, 1]).mod(tf.scalar(config.width));

    // sync and check for NaNs
    if (getEnv().debugNan === true) {
      const synced = posX.arraySync();
      for (let i = 0; i < synced.length; i++) {
        if (isNaN(synced[i])) {
          console.log("NaN at " + i);
        } else {
          // console.log("not NaN at " + i);
        }
      }
    }

    // Slice to take only the Y values then mod them by config.width to wrap around
    const posY = updatedPos
      .slice([0, 1], [-1, 1])
      .mod(tf.scalar(config.height));

    // Split velocity to clip it.
    // TODO: Clip by vel magnitude, not each direction
    // IDEAS: Add some other weird conditions(Perhaps this could tune the art)
    //
    // Cap Vels
    const velX = updatedVel
      .slice([0, 0], [-1, 1])
      .clipByValue(-config.maximumVelocity, config.maximumVelocity);

    const velY = updatedVel
      .slice([0, 1], [-1, 1])
      .clipByValue(-config.maximumVelocity, config.maximumVelocity);

    // Rejoin the X and Y fields
    const updateVelCapped = velX.concat(velY, 1);
    const updatePosWrapped = posX.concat(posY, 1);

    const updatePosReset = randomReset(updatePosWrapped, config);
    // pt('r', updatePosReset);

    // debug
    // debug
    // console.log("base");
    // pt("p", pos);

    // console.log("update, slice, cap");
    // pt("p", posX);
    // pt("v", velX);

    // debug
    // console.log("reset");
    // pt("p", updatePosReset);
    // pt("v", updateVelCapped);

    return { newP: updatePosReset, newV: updateVelCapped };
  });
}

// a.where(b, c)
//
// a = original
// b = randomParticle Positions
// c = randomBooleans distributed with false rate = config.resetRate
export function randomReset(originalTensor, config) {
  const count = originalTensor.shape[0];

  return tf.tidy(() => {
    // Make random X's and Y's and use a single tuple for easy concat,
    // NOTE: You will see a lot of singular tuples. I've found a couple patterns
    //       that work nice this way and avoid too much intermediate tensor reshaping.
    //       Might keep doing this.
    const { resetRate, width, height, density } = config;

    // Generate a distribution of random booleans.
    const randomBooleans = tf.less(
      tf.randomUniform([count, 1]),
      tf.scalar(1 - config.resetRate)
    );

    // TODO: This might be able to be done with another combine operation
    //       Not having to use the singular tuples, but I'm deving offline.

    const randomXs = tf.randomUniform([count, 1], 0, width);
    const randomYs = tf.randomUniform([count, 1], 0, height);

    // Split Positions into x,y
    const posX = originalTensor.slice([0, 0], [-1, 1]);
    const posY = originalTensor.slice([0, 1], [-1, 1]);

    const resetX = tf.where(randomBooleans, posX, randomXs);
    const resetY = tf.where(randomBooleans, posY, randomYs);

    const resetPos = resetX.concat(resetY, 1);

    // pt("rp", originalTensor);

    return resetPos;
  });
}

// a.where(b, c)
//
// a = original
// b = randomParticle Positions
// c = randomBooleans distributed with false rate = config.resetRate
export function randomReset2(originalTensor, config) {
  return tf.tidy(() => {
    const { resetRate, width, height, density } = config;

    // Generate a distribution of random booleans.
    const randomBooleans = tf.less(
      tf.randomUniform([config.particleCount, 1]),
      tf.scalar(config.resetRate)
    );

    // Make random X's and Y's and use a single tuple for easy concat,
    // TODO: This might be able to be done with another combine operation
    //       Not having to use the singular tuples, but I'm deving offline.
    const randomXs = tf.randomUniform([config.particleCount, 1]);
    const randomYs = tf.randomUniform([config.particleCount, 1]);

    // Split Positions into x,y
    const posX = originalTensor.slice([0, 0], [-1, 1]);
    const posY = originalTensor.slice([0, 1], [-1, 1]);

    // Where doesn't currently support gradients in the version I'm working  on.
    // Use negation and addition to do it instead.
    //
    // The first negation terms 0s the revelvant cells the second adds the random value in
    // x + boolean*x*-1 + boolean*randomX
    //
    const randomBinaryFlags = tf.cast(randomBooleans, "float32");

    const resetX = posX.add(
      randomBinaryFlags.neg().mul(posX).add(randomBinaryFlags.mul(randomXs))
    );
    const resetY = posY.add(
      randomBinaryFlags.neg().mul(posY).add(randomBinaryFlags.mul(randomYs))
    );

    const resetPos = resetX.concat(resetY, 1);

    return resetPos;
  });
}

// print tensor
function pt(name, tensor) {
  console.log(name);
  tensor.print(true);
}
