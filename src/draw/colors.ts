import { labConversionFunctions, rotateAB, skewL } from "../utils/color";

const magenta = [255, 0, 255];
const teal = [0, 128, 128];
const aquamarine = [127, 255, 212];

// light and dark with lab
const magnetaSet = [skewL(magenta, 1.4), skewL(magenta, 0.8)];
const tealSet = [skewL(teal, 1.4), skewL(teal, 0.8)];
const aquamarineSet = [skewL(aquamarine, 1.4), skewL(aquamarine, 0.8)];

export const pointColorSet = [magnetaSet, tealSet, aquamarineSet];

// slight rotate and shift to something low for the light color
const modelMagenta = skewL(rotateAB(magenta, 0.06), 2.8);
const modelTeal = skewL(rotateAB(teal, 0.03), 1.8);
const modelAquamarine = skewL(rotateAB(aquamarine, -0.04), 2.2);

export const modelColorSet = [modelMagenta, modelTeal, modelAquamarine];