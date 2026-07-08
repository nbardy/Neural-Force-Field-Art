export interface Camera3D {
  name: string;
  promptSuffix: string;
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
    eye: [0, H + 1.6, 0],
    target: [0, 0, 0],
    up: [0, 0, -1],
  },
  {
    name: "front",
    promptSuffix: "a front-facing camera angle",
    eye: [0, 0, R],
    target: [0, 0, 0],
  },
  {
    name: "right",
    promptSuffix: "a camera angle from the right side",
    eye: [R, 0, 0],
    target: [0, 0, 0],
  },
  {
    name: "back",
    promptSuffix: "a camera angle from behind",
    eye: [0, 0, -R],
    target: [0, 0, 0],
  },
  {
    name: "left",
    promptSuffix: "a camera angle from the left side",
    eye: [-R, 0, 0],
    target: [0, 0, 0],
  },
  {
    name: "front-left-high",
    promptSuffix: "an elevated 45 degree camera angle from the front left looking down",
    eye: [-R * 0.72, H, R * 0.72],
    target: [0, 0, 0],
  },
  {
    name: "front-right-high",
    promptSuffix: "an elevated 45 degree camera angle from the front right looking down",
    eye: [R * 0.72, H, R * 0.72],
    target: [0, 0, 0],
  },
  {
    name: "back-right-low",
    promptSuffix: "a low 45 degree camera angle from the rear right looking up",
    eye: [R * 0.72, L, -R * 0.72],
    target: [0, 0, 0],
  },
  {
    name: "back-left-low",
    promptSuffix: "a low 45 degree camera angle from the rear left looking up",
    eye: [-R * 0.72, L, -R * 0.72],
    target: [0, 0, 0],
  },
];

export const BLACK_BACKGROUND_PROMPT = "on a black background";
export const CENTERED_BLACK_BACKGROUND_PROMPT = "centered on a black background";
export type BackgroundPromptMode = "none" | "black" | "centered";
export type ViewPromptMode = "camera" | "same" | "coarse";
export type CameraFramingMode = "normal" | "zoom_out";
export type Grid9PromptMode = "contact_sheet" | "literal" | "literal_v2" | "same";

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
  return buildBasePrompt(`${base.trim() || "a photo of a cat"}, ${camera.promptSuffix}`, backgroundMode);
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

export function camerasForFraming(mode: CameraFramingMode): Camera3D[] {
  if (mode !== "zoom_out") return DEFAULT_3D_CAMERAS;
  return DEFAULT_3D_CAMERAS.map((camera) => ({
    ...camera,
    eye: [camera.eye[0] * 1.25, camera.eye[1] * 1.25, camera.eye[2] * 1.25],
    fovYDeg: Math.max(camera.fovYDeg ?? 50, 56),
  }));
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
