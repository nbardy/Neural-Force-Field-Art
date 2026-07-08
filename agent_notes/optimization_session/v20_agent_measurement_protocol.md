# v20 Agent Measurement Protocol

Date: 2026-07-08

Scope: documentation-only protocol for measuring future CLIP shader forks in
`/Users/nicholasbardy/git/Neural-Force-Field-Art`. I inspected the current
Bun/WebGPU profiling tools, the integrated splat benchmarks, the experiment fork
conventions, and the browser trace helper. I did not edit runtime source code.

## Goal

v20 should separate a real CLIP shader win from GPU contention, queue noise, or
browser scheduling noise.

Use three layers, in this order:

1. Isolated CLIP GPU timestamps: prove the targeted shader family got faster.
2. Integrated optimizer timestamps and wall time: prove the faster shader still
   matters inside the real splat step.
3. Chrome/Dawn trace: use only when queue, browser, compilation, or page
   scheduling behavior is the question.

Do not run Bun benches, Chrome traces, video, other browser tabs, or other GPU
workloads at the same time.

## Current Tools

- `tools/clip/dispatch_profile.ts`
  - `TIMESTAMP=1` uses WebGPU timestamp queries when supported.
  - `CSV=1` emits per-dispatch rows for grouping and paired-trial analysis.
  - This is the primary CLIP shader attribution tool.
- `tools/clip/batch_major_train_bench.ts`
  - End-to-end CLIP train path with parity checks.
  - Useful correctness and wall sanity check, not per-shader attribution.
- `tools/clip/batch_major_train_matrix.ts`
  - Sequential A/B runner for CLIP train wall time.
  - Reverses config order by trial, which helps order bias.
- `tools/splat3d/step_bench.ts`
  - Integrated raster -> CLIP -> raster backward -> Adam -> display benchmark.
  - `TIMESTAMP=1` timestamps the actual measured passes for profile attribution.
  - `normal step avg` remains wall-clock.
- `tools/splat3d/step_matrix.ts`
  - Sequential A/B runner for integrated steps.
  - Best promotion gate when the variant is expressible as matrix tokens.
- `tools/webgpu_trace.mjs`
  - Puppeteer/Chrome helper for WebGPU/Dawn traces.
  - Shows browser, Dawn, queue, command-buffer, and scheduling behavior.
  - It is not the numeric source of truth for shader duration.

## Required Fork Setup

Use this folder shape for every v20 measurement attempt:

```bash
FORK=experiments/clip_forks/v20_<short_slug>
DAY=$(date +%F)
OUT="$FORK/results/$DAY"
mkdir -p "$OUT"/{metadata,correctness,dispatch,integrated,trace}
```

Store:

- `README.md`: hypothesis, env gates, decision, headline table.
- `snapshot/`: source files copied before the fork's code edits.
- `results/YYYY-MM-DD/metadata/`: git head/status, adapter, exact env.
- `results/YYYY-MM-DD/correctness/`: correctness gate stdout.
- `results/YYYY-MM-DD/dispatch/`: isolated timestamp CSVs and summaries.
- `results/YYYY-MM-DD/integrated/`: `step_matrix` JSON/stdout results.
- `results/YYYY-MM-DD/trace/`: Chrome trace summaries, screenshots, and only
  compressed raw traces if the file is small enough to keep.

Raw Chrome trace JSONs can be huge. Default to saving them under `/tmp` and
commit a trace manifest with path, file size, command, adapter probe, before/after
state, and screenshot path. If the trace is decisive and reasonably sized,
archive `webgpu_trace_*.json.gz` under `results/YYYY-MM-DD/trace/`.

## Fill These Two Fork-Specific Values

Set the exact variant env vars and the matching `step_matrix` tokens once:

```bash
# Example only. Replace with the candidate's real switches.
VARIANT_ENV=(PW_TILE_VARIANT=rect8x16 PW_TILE_STEPS=57,59)
VARIANT_TOKENS="pwrect:pwsteps57-59"
```

If the fork does not have `step_matrix` tokens yet, do not promote from
ad-hoc wall timing. Add matrix-token plumbing or run the manual paired
`step_bench` fallback and record that limitation in the fork README.

## Exact Command Order

### 0. Metadata

```bash
git rev-parse HEAD | tee "$OUT/metadata/git_head.txt"
git status --short | tee "$OUT/metadata/git_status.txt"

bun -e 'import { setupGlobals } from "bun-webgpu"; setupGlobals(); const a = await navigator.gpu.requestAdapter(); console.log(JSON.stringify({adapter: a?.info ?? null, features: a ? Array.from(a.features).sort() : []}, null, 2));' \
  | tee "$OUT/metadata/adapter.json"
```

If `timestamp-query` is absent, do not make shader-win claims. Fall back to
wall-only scouting and mark the result inconclusive.

### 1. Correctness

Run correctness before timing. For forward-only variants, the train forward
gate is still required because the optimizer uses `plan_train.json`.

```bash
env "${VARIANT_ENV[@]}" PLAN=plan_train.json BENCH_RUNS=1 \
  bun tools/clip/fused_test.ts \
  | tee "$OUT/correctness/fused_train_variant.txt"

env "${VARIANT_ENV[@]}" \
  bun tools/clip/bwd_test.ts \
  | tee "$OUT/correctness/bwd_variant.txt"

env "${VARIANT_ENV[@]}" STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 BATCH=3 RUNS=1 WARMUP=1 \
  bun tools/clip/batch_major_train_bench.ts \
  | tee "$OUT/correctness/batch_major_train_variant.txt"
```

If the variant intentionally changes precision or math, add the relevant
gradient/quality gate before timing. A speedup cannot override a failed
`dL/dimage` gate.

### 2. Static Context

```bash
env BATCH=3 TOP=16 OUT="$OUT/dispatch/pointwise_report_b3.md" \
  bun tools/clip/pointwise_report.ts \
  | tee "$OUT/dispatch/pointwise_report_b3.stdout.txt"
```

This does not prove speed. It records which shapes and groups the candidate is
supposed to affect.

### 3. Isolated CLIP Dispatch Timestamps

Use five paired trials. Each profiler run uses `RUNS=7 WARMUP=3`; the tool
reports medians per dispatch. Alternate order to reduce thermal/order bias.

```bash
BASE_DISPATCH_ENV=(TIMESTAMP=1 CSV=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=7 WARMUP=3)

for t in 0 1 2 3 4; do
  if [ $((t % 2)) -eq 0 ]; then order=(base variant); else order=(variant base); fi
  for cfg in "${order[@]}"; do
    if [ "$cfg" = base ]; then
      env "${BASE_DISPATCH_ENV[@]}" \
        bun tools/clip/dispatch_profile.ts \
        > "$OUT/dispatch/dispatch_base_t${t}.csv"
    else
      env "${BASE_DISPATCH_ENV[@]}" "${VARIANT_ENV[@]}" \
        bun tools/clip/dispatch_profile.ts \
        > "$OUT/dispatch/dispatch_variant_t${t}.csv"
    fi
    sleep 10
  done
done

env TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=7 WARMUP=3 \
  bun tools/clip/dispatch_profile.ts \
  | tee "$OUT/dispatch/dispatch_base_summary.txt"

env TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=7 WARMUP=3 "${VARIANT_ENV[@]}" \
  bun tools/clip/dispatch_profile.ts \
  | tee "$OUT/dispatch/dispatch_variant_summary.txt"
```

Required readout:

- Compare median by dispatch label and by group, not only total isolated sum.
- A real shader win should improve the targeted labels/groups in at least
  four of five paired trials.
- Require either `>= 5%` targeted-group median improvement or `>= 1 ms` absolute
  targeted-group improvement. Smaller moves are noise unless repeated across
  later integrated gates.
- Reject or keep gated if non-target groups move by the same percentage. That
  pattern usually indicates GPU state, thermal drift, or contention rather than
  a local shader win.

The isolated total is a ranking sum. It is not the full optimizer CLIP runtime.

### 4. Integrated Wall-Time Promotion Gate

Run this after isolated timestamps show a plausible win. Use seven trials and
`RUNS=7 WARMUP=5` for the promotion wall gate.

```bash
env JSON=1 TRIALS=7 RUNS=7 WARMUP=5 \
  CONFIGS="base=3:3,candidate=3:3:${VARIANT_TOKENS}" \
  bun tools/splat3d/step_matrix.ts \
  > "$OUT/integrated/step_matrix_wall_3v3.json"

env TRIALS=7 RUNS=7 WARMUP=5 \
  CONFIGS="base=3:3,candidate=3:3:${VARIANT_TOKENS}" \
  bun tools/splat3d/step_matrix.ts \
  | tee "$OUT/integrated/step_matrix_wall_3v3.txt"
```

If the fork targets the current grid/direct-raster path, add the matching
schedule as a second wall gate:

```bash
env JSON=1 TRIALS=7 RUNS=7 WARMUP=5 \
  CONFIGS="base_grid=9:3:grid9:directgrid,candidate_grid=9:3:grid9:directgrid:${VARIANT_TOKENS}" \
  bun tools/splat3d/step_matrix.ts \
  > "$OUT/integrated/step_matrix_wall_grid80.json"
```

Manual fallback when matrix tokens do not exist yet:

```bash
BASE_STEP_ENV=(CLIP_BATCH=3 VIEWS=3 RUNS=7 WARMUP=5)

for t in 0 1 2 3 4 5 6; do
  if [ $((t % 2)) -eq 0 ]; then order=(base variant); else order=(variant base); fi
  for cfg in "${order[@]}"; do
    if [ "$cfg" = base ]; then
      env "${BASE_STEP_ENV[@]}" \
        bun tools/splat3d/step_bench.ts \
        | tee "$OUT/integrated/step_bench_wall_base_t${t}.txt"
    else
      env "${BASE_STEP_ENV[@]}" "${VARIANT_ENV[@]}" \
        bun tools/splat3d/step_bench.ts \
        | tee "$OUT/integrated/step_bench_wall_variant_t${t}.txt"
    fi
    sleep 10
  done
done
```

This fallback is acceptable for early scouting, but the fork should grow
`step_matrix` tokens before promotion so later agents can rerun the exact A/B
gate without hand-parsing stdout files.

Promotion rule:

- `normal` wall median must improve in the same direction as the isolated CLIP
  timestamp result.
- Prefer `>= 3%` wall median improvement or `>= 2 ms` absolute improvement for
  this app. Below that, keep the fork gated unless the change is clearly needed
  for a larger stack.
- `clip` profile median should improve more than raster/display/Adam. If raster
  or display moves by the same amount, assume contention until proven otherwise.

### 5. Integrated Timestamp Attribution

Run this after the wall gate, not instead of it. It answers whether the real
optimizer profile attributes the change to CLIP.

```bash
env TIMESTAMP=1 JSON=1 TRIALS=5 RUNS=5 WARMUP=3 \
  CONFIGS="base=3:3,candidate=3:3:${VARIANT_TOKENS}" \
  bun tools/splat3d/step_matrix.ts \
  > "$OUT/integrated/step_matrix_timestamp_3v3.json"

env TIMESTAMP=1 TRIALS=5 RUNS=5 WARMUP=3 \
  CONFIGS="base=3:3,candidate=3:3:${VARIANT_TOKENS}" \
  bun tools/splat3d/step_matrix.ts \
  | tee "$OUT/integrated/step_matrix_timestamp_3v3.txt"
```

Interpret the `profile` fields here as GPU timestamp attribution for measured
passes. `normal` is still wall-clock. `profileTotal` can differ from
`normal step avg`; do not force them to reconcile exactly.

## Timestamp vs Wall-Clock Interpretation

Use this table when writing the fork decision:

| Signal | Means | Does not mean |
| --- | --- | --- |
| `dispatch_profile.ts TIMESTAMP=1` | Isolated GPU execution time for one generated CLIP dispatch at a time. Best source for shader-family attribution. | Full CLIP runtime, app wall time, hardware counters, or browser behavior. |
| `step_bench.ts TIMESTAMP=1` profile | GPU timestamp attribution for the real optimizer passes being profiled. Best integrated CLIP-vs-raster split. | Promotion by itself; the normal app path may still be wall-limited. |
| `step_matrix.ts` without `TIMESTAMP=1` | Sequential wall-clock A/B gate for the real optimizer schedule. Best promotion gate. | Proof that the shader itself got faster. It includes CPU, queue waits, synchronization, thermals, and other GPU work. |
| Chrome trace | Browser/Dawn/queue scheduling evidence. Good for gaps, compilation, main-thread stalls, throttling, and page path sanity. | Reliable per-shader milliseconds, memory bandwidth, occupancy, cache misses, or register pressure. |

Classify results this way:

- Real shader win: targeted isolated timestamp groups improve, integrated CLIP
  timestamp improves, and wall median improves in the same direction.
- GPU contention or thermal noise: wall moves but CLIP timestamps are flat, all
  groups move together, or paired-trial order changes the winner.
- Integrated bottleneck shift: isolated CLIP improves but wall is flat because
  raster/display/CPU became dominant. Keep gated unless the stack needs it.
- Browser/page issue: Bun timestamps are stable but browser behavior is bad or
  inconsistent. Use Chrome trace.

## When To Use Chrome Trace

Do not use Chrome trace for every shader fork. Use it when one of these is true:

- Isolated timestamps show a win but integrated wall time is flat or worse.
- Wall time changes but integrated timestamps do not attribute the change to
  CLIP.
- You suspect queue gaps, command-buffer backpressure, shader/pipeline JIT,
  model loading, page throttling, renderer-main-thread stalls, compositor work,
  or readback/map synchronization.
- The question is browser-only behavior, not Bun/WebGPU behavior.

Do not use `tools/smoke.mjs` for performance. It is a headless compatibility
smoke path and may force or observe fallback adapter behavior depending on the
machine.

Trace command order:

```bash
npx parcel build --no-scope-hoist --no-cache --public-url ./ src/splat3d.html
node tools/splat/serve.mjs 8799
```

In a second terminal:

```bash
TRACE_OUT="$OUT/trace"
mkdir -p "$TRACE_OUT/base" "$TRACE_OUT/candidate"

GRID9=1 DIRECT_GRID=1 SETUP_TIMEOUT_MS=180000 TRACE_PROFILE_DIR=/tmp/nffa_trace_profile_base \
  node tools/webgpu_trace.mjs http://localhost:8799/dist/splat3d.html 12000 "$TRACE_OUT/base" \
  | tee "$TRACE_OUT/base/trace_stdout.txt"

GRID9=1 DIRECT_GRID=1 SETUP_TIMEOUT_MS=180000 TRACE_PROFILE_DIR=/tmp/nffa_trace_profile_candidate \
  node tools/webgpu_trace.mjs http://localhost:8799/dist/splat3d.html 12000 "$TRACE_OUT/candidate" \
  | tee "$TRACE_OUT/candidate/trace_stdout.txt"
```

If the candidate requires build-time env vars, rebuild once for base and once
for candidate and record the exact build env in `trace_manifest.md`.

In the trace, look for:

- adapter and fallback status in helper output;
- app state before/after optimize;
- Dawn/WebGPU queue submits and command-buffer cadence;
- pipeline creation or shader compile events during the measured window;
- renderer main-thread or V8 stalls;
- GPU-process gaps and backpressure events;
- compositor/viz work overlapping the measurement window.

Chrome trace can explain why wall time disagrees with timestamps. It should not
overrule timestamp evidence about whether a specific CLIP shader got faster.

## README Decision Template

Each `v20_*` fork README should end with:

```text
Decision: promote / keep gated / reject

Correctness:
- fused train:
- backward / dL/dimage:
- batch-major parity:

Isolated CLIP timestamps:
- targeted groups:
- isolated total:
- paired wins:

Integrated wall:
- 3:3 normal median:
- target schedule normal median:
- profile clip/raster:

Integrated timestamps:
- CLIP batch:
- raster:
- profile total:

Trace, if used:
- trace path or manifest:
- queue/browser finding:

Interpretation:
- real shader win / contention / bottleneck shift / browser issue
```

If the evidence does not satisfy the real-shader-win pattern, keep the code path
behind its env/UI gate and record the fork as no-promote. That outcome is still
useful if it narrows the next shader hypothesis.
