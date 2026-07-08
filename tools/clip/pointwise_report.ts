/**
 * Static MobileCLIP pointwise report.
 *
 * This is a CPU-only companion to dispatch_profile.ts. It explains what the
 * generated pointwise kernels are doing from the compiled plan: counts, shapes,
 * MACs/FLOPs, current tile geometry, and approximate staged global traffic.
 *
 *   bun tools/clip/pointwise_report.ts
 *   BATCH=3 TOP=16 OUT=/tmp/pointwise_report.md bun tools/clip/pointwise_report.ts
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const PLAN_FILE = process.env.PLAN ?? "plan_train.json";
const BATCH = Math.max(1, Number(process.env.BATCH ?? 3) | 0);
const TOP = Math.max(1, Number(process.env.TOP ?? 12) | 0);
const OUT = process.env.OUT ?? "";

type ConvStep = {
  kind: "conv";
  variant: "pointwise" | "depthwise" | "general";
  name: string;
  cin: number;
  cout: number;
  k: number;
  outH: number;
  outW: number;
  src: number;
  dst: number;
  residual: number | null;
  layerScaleOff: number | null;
};

type GeluStep = {
  kind: "gelu";
  src: number;
  dst: number;
  n: number;
};

type PwBwdStep = {
  kind: "pw_bwd";
  name: string;
  cin: number;
  cout: number;
  outH: number;
  outW: number;
  dY: number;
  dX: number;
  accumulate: boolean;
};

type VisionStep = ConvStep | GeluStep | { kind: string; [key: string]: unknown };
type BwdStep = PwBwdStep | { kind: string; [key: string]: unknown };

type TrainPlan = {
  model: string;
  inputShape: [number, number, number];
  weightsFloats: number;
  slots: number[];
  steps: VisionStep[];
  backward: BwdStep[];
};

type Row = {
  phase: "forward" | "backward";
  index: number;
  label: string;
  cin: number;
  cout: number;
  h: number;
  w: number;
  p: number;
  macs: number;
  flops: number;
  stagedBytes: number;
  slotBytes: number;
  workgroups: number;
  fusesGelu: boolean;
  residual: boolean;
  accumulate: boolean;
};

type ShapeSummary = {
  key: string;
  phase: "forward" | "backward";
  count: number;
  indexes: number[];
  label: string;
  cin: number;
  cout: number;
  h: number;
  w: number;
  macs: number;
  flops: number;
  stagedBytes: number;
  slotBytes: number;
  workgroups: number;
  fusesGelu: number;
  residual: number;
  accumulate: number;
};

function isPointwise(step: VisionStep): step is ConvStep {
  return step.kind === "conv" && (step as ConvStep).variant === "pointwise";
}

function isGelu(step: VisionStep | undefined): step is GeluStep {
  return step?.kind === "gelu";
}

function isPwBwd(step: BwdStep): step is PwBwdStep {
  return step.kind === "pw_bwd";
}

function num(v: number): string {
  return Number.isInteger(v) ? String(v) : v.toFixed(2);
}

function fmtG(v: number): string {
  return (v / 1e9).toFixed(3);
}

function fmtM(v: number): string {
  return (v / 1e6).toFixed(2);
}

function fmtMiB(v: number): string {
  return (v / (1024 * 1024)).toFixed(2);
}

function sum(rows: Row[], pick: (r: Row) => number): number {
  return rows.reduce((acc, row) => acc + pick(row), 0);
}

function workgroupsFor(p: number, cout: number): number {
  // pointwiseTiledMain: [P4/8, Cout/32, 1] = [(P/4)/8, Cout/32, 1].
  return (p / 32) * (cout / 32);
}

function stagedBytesFor(p: number, cin: number, cout: number, writes: number): number {
  const macs = p * cin * cout;
  // Current tile reuses each staged activation and each staged weight across
  // 32 scalar MACs. Writes are full output slots. This is an approximation of
  // compulsory global traffic; cache behavior and extra readers are measured by
  // dispatch_profile.ts, not inferred here.
  const xReadFloats = macs / 32;
  const wReadFloats = macs / 32;
  const writeFloats = p * cout * writes;
  return 4 * (xReadFloats + wReadFloats + writeFloats);
}

function slotBytesFor(p: number, cin: number, cout: number): number {
  // Source + destination tensor footprint for one image. The train graph keeps
  // saved activations, so footprint matters even when the kernel stages tiles.
  return 4 * p * (cin + cout);
}

function forwardRows(plan: TrainPlan): Row[] {
  const rows: Row[] = [];
  for (let i = 0; i < plan.steps.length; i++) {
    const step = plan.steps[i];
    if (!isPointwise(step)) continue;
    const p = step.outH * step.outW;
    const next = plan.steps[i + 1];
    const fusesGelu =
      isGelu(next) &&
      next.src === step.dst &&
      next.n === p * step.cout &&
      step.residual === null;
    const writes = fusesGelu ? 2 : 1;
    const macs = p * step.cin * step.cout;
    rows.push({
      phase: "forward",
      index: i,
      label: `pw ${step.cin}->${step.cout} @${step.outH}x${step.outW}`,
      cin: step.cin,
      cout: step.cout,
      h: step.outH,
      w: step.outW,
      p,
      macs,
      flops: 2 * macs,
      stagedBytes: stagedBytesFor(p, step.cin, step.cout, writes),
      slotBytes: slotBytesFor(p, step.cin, step.cout),
      workgroups: workgroupsFor(p, step.cout),
      fusesGelu,
      residual: step.residual !== null,
      accumulate: false,
    });
  }
  return rows;
}

function backwardRows(plan: TrainPlan): Row[] {
  const rows: Row[] = [];
  for (let i = 0; i < plan.backward.length; i++) {
    const step = plan.backward[i];
    if (!isPwBwd(step)) continue;
    const p = step.outH * step.outW;
    const macs = p * step.cin * step.cout;
    rows.push({
      phase: "backward",
      index: i,
      label: `pw_bwd ${step.cin}->${step.cout} @${step.outH}x${step.outW}`,
      cin: step.cin,
      cout: step.cout,
      h: step.outH,
      w: step.outW,
      p,
      macs,
      flops: 2 * macs,
      stagedBytes: stagedBytesFor(p, step.cin, step.cout, 1),
      slotBytes: slotBytesFor(p, step.cin, step.cout),
      workgroups: workgroupsFor(p, step.cout),
      fusesGelu: false,
      residual: false,
      accumulate: step.accumulate,
    });
  }
  return rows;
}

function byShape(rows: Row[]): ShapeSummary[] {
  const map = new Map<string, ShapeSummary>();
  for (const row of rows) {
    const key = `${row.phase}:${row.cin}->${row.cout}@${row.h}x${row.w}`;
    let s = map.get(key);
    if (!s) {
      s = {
        key,
        phase: row.phase,
        count: 0,
        indexes: [],
        label: row.label,
        cin: row.cin,
        cout: row.cout,
        h: row.h,
        w: row.w,
        macs: 0,
        flops: 0,
        stagedBytes: 0,
        slotBytes: 0,
        workgroups: 0,
        fusesGelu: 0,
        residual: 0,
        accumulate: 0,
      };
      map.set(key, s);
    }
    s.count++;
    s.indexes.push(row.index);
    s.macs += row.macs;
    s.flops += row.flops;
    s.stagedBytes += row.stagedBytes;
    s.slotBytes += row.slotBytes;
    s.workgroups += row.workgroups;
    if (row.fusesGelu) s.fusesGelu++;
    if (row.residual) s.residual++;
    if (row.accumulate) s.accumulate++;
  }
  return [...map.values()].sort((a, b) => b.flops - a.flops);
}

function table(headers: string[], rows: string[][]): string {
  const out = [
    `| ${headers.join(" | ")} |`,
    `| ${headers.map(() => "---").join(" | ")} |`,
  ];
  for (const row of rows) out.push(`| ${row.join(" | ")} |`);
  return out.join("\n");
}

function render(plan: TrainPlan, rows: Row[]): string {
  const fwd = rows.filter((r) => r.phase === "forward");
  const bwd = rows.filter((r) => r.phase === "backward");
  const fusable = fwd.filter((r) => r.fusesGelu).length;
  const totalFlops = sum(rows, (r) => r.flops);
  const totalStaged = sum(rows, (r) => r.stagedBytes);
  const totalWorkgroups = sum(rows, (r) => r.workgroups);
  const topShapes = byShape(rows).slice(0, TOP);
  const topRows = [...rows].sort((a, b) => b.flops - a.flops).slice(0, TOP);

  const lines: string[] = [];
  lines.push("# MobileCLIP Pointwise Static Report");
  lines.push("");
  lines.push(`Plan: \`${PLAN_FILE}\` (${plan.model})`);
  lines.push(`Input resolution: \`${plan.inputShape.join("x")}\``);
  lines.push(`Batch multiplier shown: \`${BATCH}\``);
  lines.push("");
  lines.push("## What Pointwise Means Here");
  lines.push("");
  lines.push("Pointwise is a `1x1`, `groups=1` convolution. For each pixel/token position, it is a dense channel matrix multiply:");
  lines.push("");
  lines.push("```text");
  lines.push("Y[p, co] = bias[co] + sum_ci X[p, ci] * W[ci, co]");
  lines.push("```");
  lines.push("");
  lines.push("The current WGSL stores activations as channel-planar `vec4f` pixel quads and stores pointwise weights transposed as `[Cin][Cout]` so one `W4()` load gives four adjacent output channels.");
  lines.push("");
  lines.push("Current tile geometry:");
  lines.push("");
  lines.push("- workgroup size: `8 x 8 = 64` threads");
  lines.push("- tile: `8` pixel-quads by `32` output channels = `32` pixels by `32` channels");
  lines.push("- workgroup memory: `xS=256 vec4f` + `wS=256 vec4f` = `8192` bytes");
  lines.push("- each thread accumulates four `vec4f` outputs: `4` output channels by `4` pixels");
  lines.push("");
  lines.push("## Static Totals");
  lines.push("");
  lines.push(table(
    ["Metric", "Per Image", `Batch ${BATCH}`],
    [
      ["forward pointwise dispatches", num(fwd.length), num(fwd.length)],
      ["backward `pw_bwd` dispatches", num(bwd.length), num(bwd.length)],
      ["forward pointwise+GELU candidates", num(fusable), num(fusable)],
      ["pointwise FLOPs", `${fmtG(totalFlops)} GFLOP`, `${fmtG(totalFlops * BATCH)} GFLOP`],
      ["approx staged global traffic", `${fmtMiB(totalStaged)} MiB`, `${fmtMiB(totalStaged * BATCH)} MiB`],
      ["pointwise workgroups", num(totalWorkgroups), num(totalWorkgroups * BATCH)],
    ]
  ));
  lines.push("");
  lines.push("The traffic number is a lower-bound model for the current tiled kernels: staged activation reads + staged weight reads + output writes. It does not include cache misses, extra residual/scale reads, command overhead, or non-pointwise kernels.");
  lines.push("");
  lines.push("## Top Shapes By FLOPs");
  lines.push("");
  lines.push(table(
    ["Phase", "Shape", "Count", "Indexes", `FLOPs B${BATCH}`, `Traffic B${BATCH}`, "Intensity", "Flags"],
    topShapes.map((s) => {
      const flags = [
        s.fusesGelu ? `${s.fusesGelu} gelu-fuse` : "",
        s.residual ? `${s.residual} residual` : "",
        s.accumulate ? `${s.accumulate} accumulate` : "",
      ].filter(Boolean).join(", ") || "-";
      return [
        s.phase,
        `${s.cin}->${s.cout} @${s.h}x${s.w}`,
        String(s.count),
        s.indexes.join(","),
        `${fmtG(s.flops * BATCH)}G`,
        `${fmtMiB(s.stagedBytes * BATCH)}MiB`,
        `${(s.flops / Math.max(s.stagedBytes, 1)).toFixed(1)} FLOP/B`,
        flags,
      ];
    })
  ));
  lines.push("");
  lines.push("## Top Individual Dispatches By FLOPs");
  lines.push("");
  lines.push(table(
    ["Phase", "Index", "Label", `FLOPs B${BATCH}`, `Traffic B${BATCH}`, "Workgroups B", "Flags"],
    topRows.map((r) => {
      const flags = [
        r.fusesGelu ? "gelu-fuse" : "",
        r.residual ? "residual" : "",
        r.accumulate ? "accumulate" : "",
      ].filter(Boolean).join(", ") || "-";
      return [
        r.phase,
        String(r.index),
        r.label,
        `${fmtG(r.flops * BATCH)}G`,
        `${fmtMiB(r.stagedBytes * BATCH)}MiB`,
        num(r.workgroups * BATCH),
        flags,
      ];
    })
  ));
  lines.push("");
  lines.push("## Why This Can Bottleneck");
  lines.push("");
  lines.push("Pointwise is many small-to-medium matmuls rather than one huge GEMM. The arithmetic is large, but each layer is a separate specialized dispatch and the train path runs both forward and backward. Batch-major CLIP reduces outer scheduling overhead, but the normal z-batch path still repeats the same pointwise work per rendered view.");
  lines.push("");
  lines.push("Fusing pointwise+GELU helps because it removes a separate elementwise read/write and dispatch while still preserving the pre-activation needed by backward. It cannot make the whole CLIP graph one shader because later layers need barriers between dependent tensors, attention/SE/spatial kernels have different dataflow, and train mode must keep saved activations for `dL/dimage`.");
  lines.push("");
  lines.push("## Next Exact-Math Forks");
  lines.push("");
  lines.push("1. Rectangular pointwise tiles for selected hot shapes, especially late high-channel shapes.");
  lines.push("2. Split-K `pw_bwd` for low-spatial, high-channel backward layers.");
  lines.push("3. Pointwise-specific f16 storage for selected weights or hidden GELU outputs, with full input-gradient gates.");
  lines.push("4. Proxy/teacher schedules only after the exact CLIP report and timestamp profile agree that kernel work is still the limiting factor.");
  lines.push("");
  return `${lines.join("\n")}\n`;
}

const plan = JSON.parse(readFileSync(join(MODEL_DIR, PLAN_FILE), "utf8")) as TrainPlan;
const rows = [...forwardRows(plan), ...backwardRows(plan)];
const markdown = render(plan, rows);
if (OUT) writeFileSync(OUT, markdown);
process.stdout.write(markdown);
