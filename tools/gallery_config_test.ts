/**
 * Pure gallery/config gate for the promoted fresh-install recipe.
 *
 *   bun tools/gallery_config_test.ts
 */
import {
  DEFAULT_PIECE_INDEX,
  DEFAULT_PIECE_NAME,
  GALLERY,
  forceMagnitudeForDrive,
  resolveLiveGameControls,
} from "../src/main";

let failures = 0;
function ok(condition: boolean, message: string): void {
  console.log(`${condition ? "  ok  " : " FAIL "} ${message}`);
  if (!condition) failures++;
}

const promoted = GALLERY.find(
  (piece) => piece.name === "Adversary · Point · HashGrid · Curl · WTA10 · Coolwarm Raw"
);
ok(!!promoted, "promoted Point/WTA10 piece exists");
if (promoted) {
  ok(DEFAULT_PIECE_INDEX === GALLERY.findIndex((piece) => piece.name === promoted.name), "fresh default resolves to the promoted recipe");
  ok(DEFAULT_PIECE_NAME === promoted.name, "fresh default resolves by the new name");
  ok(promoted.particleCount === 190000, "particle count is 190000");
  ok(promoted.maxVelocity === 65.75, "max velocity is 65.75");
  ok(promoted.sampleRate === 9584, "sample rate is 9584");
  ok(promoted.discriminatorLearningRate === 0.00002660725059798809, "D LR is promoted");
  ok(
    promoted.colorMode?.tag === "surprise-raw" && promoted.colorMode.colormap === "coolwarm",
    "default colour mode is RAW / coolwarm"
  );
  ok(promoted.border?.tag === "reset", "border is reset");
  ok(promoted.stroke === "curl" && promoted.strokeLen === 3, "curl length is 3");
  ok(promoted.fieldArch?.encoding === "hashgrid", "field architecture is HashGrid");
  ok(
    promoted.adversary?.tag === "on" &&
      promoted.adversary.kind.tag === "wta" &&
      promoted.adversary.kind.k === 10 &&
      promoted.adversary.kind.relaxEps === 0.1 &&
      promoted.adversary.encoding.tag === "point" &&
      promoted.adversary.loss?.tag === "soft-angle",
    "game is point / WTA K10 / soft-angle"
  );
  const controls = resolveLiveGameControls(promoted, new URLSearchParams());
  ok(controls.generatorLearningRate === 0.0048, "G LR resolves from the piece");
  ok(
    controls.discriminatorLearningRate === 0.00002660725059798809,
    "D LR resolves from the piece"
  );
  ok(
    Math.abs(promoted.forceMagnitude - forceMagnitudeForDrive(0.9, 65.75, 0.97)) < 1e-12,
    "force magnitude follows the drive bound"
  );
}

const sand = GALLERY.find((piece) => piece.name === "Sand of Times");
ok(!!sand, "Sand of Times exists as a named piece");
if (sand) {
  ok(sand.particleCount === 190000 && sand.sampleRate === 9584, "Sand uses the captured counts");
  ok(sand.discriminatorLearningRate === 0.000005308844442309883, "Sand preserves the captured D LR");
  ok(sand.colorMode?.tag === "surprise-raw" && sand.colorMode.colormap === "inferno", "Sand uses RAW/inferno");
  ok(sand.fieldArch?.alpha === 0.8, "Sand preserves the 0.8 blend");
  ok(
    sand.adversary?.tag === "on" &&
      sand.adversary.encoding.tag === "pair-rotation-scale-adjusted" &&
      sand.adversary.kind.tag === "wta" &&
      sand.adversary.kind.k === 10 &&
      sand.adversary.kind.relaxEps === 0.1 &&
      sand.adversary.predictor?.hiddenUnits === 128 &&
      sand.adversary.predictor.featureDim === 64,
    "Sand uses pair / WTA10 / wider predictor"
  );
}

// These named entries predate the promotion and must remain present. Their
// numeric indices are intentionally not rewritten here; the append-only order
// is the compatibility contract.
for (const name of [
  "Adversary · Pair · HashGrid · Curl",
  "Adversary · Pair WTA K=4",
  "Adversary · RGB Families · HashGrid",
]) {
  ok(GALLERY.some((piece) => piece.name === name), `${name} remains present`);
}

if (failures) process.exit(1);
console.log("gallery config ok");
