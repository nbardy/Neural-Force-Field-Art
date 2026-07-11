import {
  buildViewPrompt,
  DEFAULT_3D_CAMERAS,
  sampleWeightedCameraIndices,
} from "../../src/splat3d/cameras";

let state = 0x517cc1b7;
const random = (): number => {
  state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
  return state / 4294967296;
};

const counts = new Uint32Array(DEFAULT_3D_CAMERAS.length);
for (let trial = 0; trial < 20_000; trial++) {
  for (const view of sampleWeightedCameraIndices(DEFAULT_3D_CAMERAS, 3, random)) counts[view]++;
}

const canonical = counts[0] + counts[1] + counts[2] + counts[4];
const oblique = counts[3] + counts[5] + counts[6] + counts[7] + counts[8];
const prompts = DEFAULT_3D_CAMERAS.slice(0, 5).map((camera) => buildViewPrompt("a cat", camera, false));

console.log(
  DEFAULT_3D_CAMERAS.map((camera, index) => `${camera.name}:${counts[index]}`).join(" ")
);
console.log(prompts.join("\n"));

if (canonical / oblique < 2.5) throw new Error("GATE FAIL: canonical camera bias is too weak");
if (!prompts[0].startsWith("a directly overhead view of a cat")) throw new Error("GATE FAIL: overhead prefix");
if (!prompts[1].startsWith("a front-on view of a cat")) throw new Error("GATE FAIL: front prefix");
if (!prompts[2].startsWith("a right-side view of a cat")) throw new Error("GATE FAIL: side prefix");
if (sampleWeightedCameraIndices(DEFAULT_3D_CAMERAS, 9, random).length !== 9) {
  throw new Error("GATE FAIL: weighted sampling must remain without replacement");
}
console.log("GATE PASS: canonical views are biased, unique, and use prefix prompts.");
