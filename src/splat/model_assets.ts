import type { TrainPlan } from "../clip/vision";
import { fetchArrayBufferWithProgress, formatProgress } from "./fetch_progress";

const HF_MODEL_BASE = "https://huggingface.co/Nbardy/nff-clip-splat-weights/resolve/main/";

export interface ClipTrainAssets {
  plan: TrainPlan;
  weights: Float32Array;
  base: string;
}

export async function loadClipTrainAssets(setStatus: (message: string) => void): Promise<ClipTrainAssets> {
  const errors: string[] = [];
  for (const base of candidateModelBases()) {
    const label = modelBaseLabel(base);
    try {
      setStatus(`fetching CLIP plan${label}...`);
      const plan = await fetchPlan(base);
      const wbuf = await fetchArrayBufferWithProgress(base + "weights_train.bin", (p) => {
        setStatus(formatProgress(`loading CLIP weights${label}`, p));
      });
      return { plan, weights: new Float32Array(wbuf), base };
    } catch (e: any) {
      errors.push(`${base}: ${e?.message ?? e}`);
    }
  }
  throw new Error(`could not load CLIP train assets:\n${errors.join("\n")}`);
}

function candidateModelBases(): string[] {
  const explicit = new URLSearchParams(location.search).get("modelBase");
  const bases: string[] = [];
  if (explicit) bases.push(withTrailingSlash(explicit));
  if (["localhost", "127.0.0.1"].includes(location.hostname)) {
    // Parcel serves /models as an HTML fallback; try it first for repo-root
    // static servers, then try the known local model server used in dev.
    bases.push("/models/mobileclip_s0/");
    bases.push(`http://${location.hostname}:8799/models/mobileclip_s0/`);
  }
  bases.push(HF_MODEL_BASE);
  return [...new Set(bases)];
}

function withTrailingSlash(value: string): string {
  return value.endsWith("/") ? value : `${value}/`;
}

function modelBaseLabel(base: string): string {
  if (base === HF_MODEL_BASE) return " from HF";
  if (base.includes(":8799/")) return " from local model server";
  if (base.startsWith("http")) return ` from ${base}`;
  return "";
}

async function fetchPlan(base: string): Promise<TrainPlan> {
  const url = base + "plan_train.json";
  const res = await fetch(url);
  if (!res.ok) throw new Error(`plan_train.json fetch ${res.status}`);
  const text = await res.text();
  const head = text.trimStart().slice(0, 80);
  if (head.startsWith("<!DOCTYPE") || head.startsWith("<html")) {
    throw new Error("plan_train.json returned HTML instead of JSON");
  }
  try {
    return JSON.parse(text) as TrainPlan;
  } catch (e: any) {
    throw new Error(`plan_train.json invalid JSON: ${e?.message ?? e}`);
  }
}
