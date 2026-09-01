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
  (piece) => piece.name === "Adversary · Point · HashGrid · Curl · WTA10"
);
ok(!!promoted, "promoted Point/WTA10 piece exists");
if (promoted) {
  ok(DEFAULT_PIECE_INDEX === GALLERY.length - 1, "fresh default is the appended piece");
  ok(DEFAULT_PIECE_NAME === promoted.name, "fresh default resolves by the new name");
  ok(promoted.particleCount === 190000, "particle count is 190000");
  ok(promoted.maxVelocity === 65.75, "max velocity is 65.75");
  ok(promoted.sampleRate === 9584, "sample rate is 9584");
  ok(promoted.discriminatorLearningRate === 0.00011208369124213449, "D LR is promoted");
  ok(promoted.border?.tag === "reset", "border is reset");
  ok(promoted.stroke === "curl" && promoted.strokeLen === 3, "curl length is 3");
  ok(promoted.fieldArch?.encoding === "hashgrid", "field architecture is HashGrid");
  ok(
    promoted.adversary?.tag === "on" &&
      promoted.adversary.kind.tag === "wta" &&
      promoted.adversary.kind.k === 10 &&
      promoted.adversary.encoding.tag === "point" &&
      promoted.adversary.loss?.tag === "soft-angle",
    "game is point / WTA K10 / soft-angle"
  );
  const controls = resolveLiveGameControls(promoted, new URLSearchParams());
  ok(controls.generatorLearningRate === 0.0048, "G LR resolves from the piece");
  ok(
    controls.discriminatorLearningRate === 0.00011208369124213449,
    "D LR resolves from the piece"
  );
  ok(
    Math.abs(promoted.forceMagnitude - forceMagnitudeForDrive(0.9, 65.75, 0.97)) < 1e-12,
    "force magnitude follows the drive bound"
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
