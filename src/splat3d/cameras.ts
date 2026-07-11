export interface Camera3D {
  name: string;
  promptSuffix: string;
  /** Natural language placed before the subject for CLIP's common canonical views. */
  promptPrefix?: string;
  /** Relative probability used by the random-without-replacement view sampler. */
  sampleWeight?: number;
  eye: [number, number, number];
  target: [number, number, number];
  up?: [number, number, number];
  fovYDeg?: number;
}

export interface PreparedCamera3D extends Camera3D {
  right: [number, number, number];
  cameraUp: [number, number, number];
  forward: [number, number, number];
  focalPx: number;
}

const R = 3.0;
const H = 1.7;
const L = -1.3;

export const DEFAULT_3D_CAMERAS: Camera3D[] = [
  {
    name: "top",
    promptSuffix: "a top-down camera angle",
    promptPrefix: "a directly overhead view of",
    sampleWeight: 3,
    eye: [0, H + 1.6, 0],
    target: [0, 0, 0],
    up: [0, 0, -1],
  },
  {
    name: "front",
    promptSuffix: "a front-facing camera angle",
    promptPrefix: "a front-on view of",
    sampleWeight: 4,
    eye: [0, 0, R],
    target: [0, 0, 0],
  },
  {
    name: "right",
    promptSuffix: "a camera angle from the right side",
    promptPrefix: "a right-side view of",
    sampleWeight: 3,
    eye: [R, 0, 0],
    target: [0, 0, 0],
  },
  {
    name: "back",
    promptSuffix: "a camera angle from behind",
    promptPrefix: "a rear view of",
    sampleWeight: 0.75,
    eye: [0, 0, -R],
    target: [0, 0, 0],
  },
  {
    name: "left",
    promptSuffix: "a camera angle from the left side",
    promptPrefix: "a left-side view of",
    sampleWeight: 3,
    eye: [-R, 0, 0],
    target: [0, 0, 0],
  },
  {
    name: "front-left-high",
    promptSuffix: "an elevated 45 degree camera angle from the front left looking down",
    eye: [-R * 0.72, H, R * 0.72],
    target: [0, 0, 0],
    sampleWeight: 1,
  },
  {
    name: "front-right-high",
    promptSuffix: "an elevated 45 degree camera angle from the front right looking down",
    eye: [R * 0.72, H, R * 0.72],
    target: [0, 0, 0],
    sampleWeight: 1,
  },
  {
    name: "back-right-low",
    promptSuffix: "a low 45 degree camera angle from the rear right looking up",
    eye: [R * 0.72, L, -R * 0.72],
    target: [0, 0, 0],
    sampleWeight: 0.75,
  },
  {
    name: "back-left-low",
    promptSuffix: "a low 45 degree camera angle from the rear left looking up",
    eye: [-R * 0.72, L, -R * 0.72],
    target: [0, 0, 0],
    sampleWeight: 0.75,
  },
];

export const BLACK_BACKGROUND_PROMPT = "on a black background";
export const CENTERED_BLACK_BACKGROUND_PROMPT = "centered on a black background";
export type BackgroundPromptMode = "none" | "black" | "centered";
export type ViewPromptMode = "camera" | "same" | "coarse";
export type CameraFramingMode = "normal" | "zoom_out";
export type Grid9PromptMode = "contact_sheet" | "literal" | "literal_v2" | "same";
export const FIXED_GRID_CAMERA_COUNT = 9;
export const DUAL_GRID_RANDOM_START = 9;
export const DUAL_GRID_ZOOM_START = 18;
export const DUAL_GRID_CAMERA_COUNT = 27;

export function normalizeBackgroundPromptMode(mode: boolean | BackgroundPromptMode = true): BackgroundPromptMode {
  if (mode === true) return "black";
  if (mode === false) return "none";
  return mode;
}

export function buildBasePrompt(base: string, backgroundMode: boolean | BackgroundPromptMode = true): string {
  const text = base.trim() || "a photo of a cat";
  const mode = normalizeBackgroundPromptMode(backgroundMode);
  if (mode === "none" || /\bblack background\b/i.test(text)) return text;
  const phrase = mode === "centered" ? CENTERED_BLACK_BACKGROUND_PROMPT : BLACK_BACKGROUND_PROMPT;
  return `${text}, ${phrase}`;
}

export function buildViewPrompt(base: string, camera: Camera3D, backgroundMode: boolean | BackgroundPromptMode = true): string {
  const subject = base.trim() || "a photo of a cat";
  const view = camera.promptPrefix ? `${camera.promptPrefix} ${subject}` : `${subject}, ${camera.promptSuffix}`;
  return buildBasePrompt(view, backgroundMode);
}

export function sampleWeightedCameraIndices(
  cameras: Camera3D[],
  count: number,
  random: () => number
): number[] {
  const available = cameras.map((_camera, index) => index);
  const output: number[] = [];
  const target = Math.max(0, Math.min(available.length, count | 0));
  while (output.length < target) {
    let total = 0;
    for (const index of available) total += Math.max(0, cameras[index].sampleWeight ?? 1);
    if (!(total > 0)) {
      output.push(...available.splice(0, target - output.length));
      break;
    }
    let cursor = random() * total;
    let slot = available.length - 1;
    for (let i = 0; i < available.length; i++) {
      cursor -= Math.max(0, cameras[available[i]].sampleWeight ?? 1);
      if (cursor < 0) {
        slot = i;
        break;
      }
    }
    output.push(available[slot]);
    available.splice(slot, 1);
  }
  return output;
}

export function buildCoarseViewPrompt(
  base: string,
  camera: Camera3D,
  backgroundMode: boolean | BackgroundPromptMode = true
): string {
  return buildBasePrompt(`${base.trim() || "a photo of a cat"}, ${coarsePromptSuffix(camera)}`, backgroundMode);
}

export function buildGrid9Prompt(
  base: string,
  backgroundMode: boolean | BackgroundPromptMode = true,
  mode: Grid9PromptMode = "contact_sheet"
): string {
  if (mode === "same") return buildBasePrompt(base, backgroundMode);
  const text = base.trim() || "a photo of a cat";
  const bgMode = normalizeBackgroundPromptMode(backgroundMode);
  const bg = bgMode !== "none" && !/\bblack background\b/i.test(text) ? ", centered on a black background" : "";
  const viewList =
    "top-down view, front-facing view, right side view, rear view, left side view, " +
    "elevated front-left view looking down, elevated front-right view looking down, " +
    "low rear-right view looking up, and low rear-left view looking up";
  if (mode === "literal_v2") {
    return `a grid of 9 different camera angles of the same object, the object is centered, and the object is ${text}${bg}`;
  }
  if (mode === "literal") {
    return (
      `a 3x3 grid showing ${text} from 9 different camera angles${bg}. ` +
      `The 9 panels show the same subject in reading order: ${viewList}`
    );
  }
  return (
    `a 3x3 image grid showing the same subject, ${text}, from nine different camera angles${bg}: ` +
    viewList
  );
}

export function buildRandomGrid9Prompt(base: string, backgroundMode: boolean | BackgroundPromptMode = true): string {
  const text = base.trim() || "a photo of a cat";
  const bgMode = normalizeBackgroundPromptMode(backgroundMode);
  const bg = bgMode !== "none" && !/\bblack background\b/i.test(text) ? ", centered on a black background" : "";
  return `a 3x3 grid of nine varied camera views of the same object, the object is centered, and the object is ${text}${bg}`;
}

export function buildZoomPrompt(base: string, backgroundMode: boolean | BackgroundPromptMode = true): string {
  const text = base.trim() || "a photo of a cat";
  return buildBasePrompt(`a zoomed-in close-up view of ${text}`, backgroundMode);
}

export function camerasForFraming(mode: CameraFramingMode): Camera3D[] {
  return applyFraming(DEFAULT_3D_CAMERAS, mode);
}

export function dualGridCamerasForFraming(mode: CameraFramingMode): Camera3D[] {
  return [
    ...applyFraming(DEFAULT_3D_CAMERAS, mode),
    ...applyFraming(randomOrbitCameras("random"), mode),
    ...applyFraming(randomOrbitCameras("zoom"), mode),
  ];
}

function applyFraming(cameras: Camera3D[], mode: CameraFramingMode): Camera3D[] {
  if (mode !== "zoom_out") return cameras.map((camera) => ({ ...camera }));
  return cameras.map((camera) => ({
    ...camera,
    eye: [camera.eye[0] * 1.25, camera.eye[1] * 1.25, camera.eye[2] * 1.25],
    fovYDeg: Math.max(camera.fovYDeg ?? 50, 56),
  }));
}

function randomOrbitCameras(kind: "random" | "zoom"): Camera3D[] {
  const yawDeg = [18, 63, 111, 157, 206, 249, 292, 334, 38];
  const pitchDeg = [16, -10, 29, -24, 8, 34, -32, 21, -18];
  const radius = kind === "zoom" ? 2.15 : 3.15;
  const fovYDeg = kind === "zoom" ? 34 : 50;
  return yawDeg.map((yaw, i) => {
    const pitch = pitchDeg[i];
    const yawRad = (yaw * Math.PI) / 180;
    const pitchRad = (pitch * Math.PI) / 180;
    const h = radius * Math.cos(pitchRad);
    const eye: [number, number, number] = [
      h * Math.sin(yawRad),
      radius * Math.sin(pitchRad),
      h * Math.cos(yawRad),
    ];
    return {
      name: `${kind}-${String(i + 1).padStart(2, "0")}`,
      promptSuffix:
        kind === "zoom"
          ? `a zoomed-in close-up camera angle ${orbitPhrase(yaw, pitch)}`
          : `a varied camera angle ${orbitPhrase(yaw, pitch)}`,
      eye,
      target: [0, 0, 0],
      fovYDeg,
    };
  });
}

function orbitPhrase(yaw: number, pitch: number): string {
  const y = ((yaw % 360) + 360) % 360;
  const side =
    y < 22.5 || y >= 337.5
      ? "from the front"
      : y < 67.5
        ? "from the front right"
        : y < 112.5
          ? "from the right side"
          : y < 157.5
            ? "from the rear right"
            : y < 202.5
              ? "from behind"
              : y < 247.5
                ? "from the rear left"
                : y < 292.5
                  ? "from the left side"
                  : "from the front left";
  if (pitch > 12) return `${side}, looking slightly down`;
  if (pitch < -12) return `${side}, looking slightly up`;
  return side;
}

export function prepareCamera(camera: Camera3D, side: number): PreparedCamera3D {
  const forward = normalize(sub(camera.target, camera.eye));
  const upHint = camera.up ?? [0, 1, 0];
  let right = normalize(cross(forward, upHint));
  if (length(right) < 1e-5) right = normalize(cross(forward, [0, 0, 1]));
  const cameraUp = normalize(cross(right, forward));
  const fovY = ((camera.fovYDeg ?? 50) * Math.PI) / 180;
  const focalPx = 0.5 * side / Math.tan(0.5 * fovY);
  return { ...camera, right, cameraUp, forward, focalPx };
}

function sub(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function cross(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function length(v: [number, number, number]): number {
  return Math.hypot(v[0], v[1], v[2]);
}

function normalize(v: [number, number, number]): [number, number, number] {
  const inv = 1 / Math.max(length(v), 1e-9);
  return [v[0] * inv, v[1] * inv, v[2] * inv];
}

function coarsePromptSuffix(camera: Camera3D): string {
  switch (camera.name) {
    case "top":
      return "a top-down view";
    case "front":
      return "a front view";
    case "back":
      return "a back view";
    case "left":
    case "right":
      return "a side view";
    default:
      return camera.eye[1] >= 0 ? "an elevated side view looking down" : "a low side view looking up";
  }
}
