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

export function buildBasePrompt(base: string, includeBlackBackground = true): string {
  const text = base.trim() || "a photo of a cat";
  if (!includeBlackBackground || /\bblack background\b/i.test(text)) return text;
  return `${text}, ${BLACK_BACKGROUND_PROMPT}`;
}

export function buildViewPrompt(base: string, camera: Camera3D, includeBlackBackground = true): string {
  return buildBasePrompt(`${base.trim() || "a photo of a cat"}, ${camera.promptSuffix}`, includeBlackBackground);
}

export function buildGrid9Prompt(base: string, includeBlackBackground = true): string {
  const text = base.trim() || "a photo of a cat";
  return buildBasePrompt(
    `${text}, a 3x3 contact sheet of the same subject from nine camera angles: ` +
      "top-down view, front-facing view, right side view, rear view, left side view, " +
      "elevated front-left view looking down, elevated front-right view looking down, " +
      "low rear-right view looking up, and low rear-left view looking up",
    includeBlackBackground
  );
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
