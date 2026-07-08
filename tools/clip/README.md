# tools/clip — fused WGSL MobileCLIP-S0 vision encoder

Goal (HANDOFF.md §CLIP): score the particle canvas against a text prompt with
a CLIP-style loss, entirely on-GPU, so the force field can optimize toward a
prompt. Standard runtimes can't do this in a real-time loop — ORT-WebGPU
issues hundreds of JS-driven dispatches per forward and cannot backprop — so
the vision encoder is hand-ported to fused WGSL (99 dispatches, one submit,
weights GPU-resident).

Model: **MobileCLIP-S0** (Apple), image encoder = MCi0, a *reparameterized*
FastViT — at inference it's a clean spine of convs + 2 attention blocks,
11.35M params, 2.385 GMACs at 256×256. ONNX export from
`Xenova/mobileclip_s0` (fp32 `vision_model.onnx`, 45.5 MB). Preprocessing is
just RGB/255 at 256×256 — **no mean/std normalization** (see
`preprocessor_config.json`), which keeps the future canvas→tensor step a
single trivial kernel.

## Pipeline (all outputs under models/mobileclip_s0/, gitignored, regenerable)

```bash
# 0. one-time: fetch model (see HANDOFF §CLIP for URLs) into models/mobileclip_s0/

# 1. ORT CPU baseline + deterministic fixture (input + golden embedding)
node tools/clip/onnx_forward.mjs                # ~60 ms/forward @ 4 threads

# 2. κ — compile the ONNX graph into the typed plan + packed weights
uv run --with onnx --with numpy python tools/clip/compile_plan.py
#    → plan.json (steps: conv/se/attn_core/head + slot allocation)
#    → weights.bin (one 16B-aligned f32 blob; BN + q-scale folded into qkv)

# 3. per-step goldens from ORT (bisection oracle)
uv run --with onnx --with numpy --with onnxruntime python tools/clip/dump_refs.py

# 4. verify + bench on a real Metal adapter (bun-webgpu, headless)
bun tools/clip/fused_test.ts                    # per-step relL∞ + cosine + bench
FAST=1 bun tools/clip/fused_test.ts             # skip per-step, embedding + bench only
```

`tools/clip/graph_dump.py` prints the op inventory (debugging aid for κ).

## Results (Apple metal-3, shared GPU ±30%)

- **All 99 steps pass** at relL∞ < 2e-3 (worst observed ~4e-5); final
  embedding **cosine = 1.000000** vs ORT (relL∞ ~1e-6). The GELU is the
  exact-erf form (A&S 7.1.26 poly, |err| ≤ 1.5e-7), matching ONNX.
- **≈6.2–6.6 ms/forward** steady-state, **≈0.15 ms CPU** encode+submit.
  Measured baselines, same model + fixture, verified cosine 1.000000 each:
  | runtime | ms/forward |
  | --- | --- |
  | ORT CPU (onnxruntime-node 1.27, 4 threads) | ~60 |
  | ORT WebGPU (ort-web 1.27, headless Chrome, same Metal adapter, warmed — `node tools/clip/ort_web_bench.mjs`) | ~11.3–12 mean · 9.5 min |
  | **fused WGSL (this)** | **~6.2–6.6** |
  Wall-clock ~1.8× vs ORT-WebGPU; the bigger structural wins are CPU cost
  (~0.15 ms vs a JS-driven per-op dispatch loop), GPU-resident I/O (no
  tensor upload/readback per frame — the canvas and the gradient stay on
  device), and a graph you can hand-write a backward for. Optimization
  history: naive kernels 16.2 → tiled pointwise + spatial-conv 6.7 →
  attention-as-pointwise ≈6.2–6.6.

## Design (mirrors the advect/train kernel philosophy)

- `tools/clip/compile_plan.py` (κ): the ONLY place ONNX is understood.
  Pattern-matches the 518-node graph into FOUR canonical step kinds — `conv`
  (pointwise/depthwise/general), `se`, `attn_core`, `head` — and THROWS on
  any unmatched node. Compile-time folds: BN prenorm → qkv weights+bias,
  q·(1/√d) → q columns, GELU/layer-scale/residual → conv step flags.
  Slot allocator frees a tensor after its last consumer; the INPUT slot is
  pinned (phantom consumer) so `run()` is repeatable — reusing it cost a
  debugging session (second forward read stale activations as the image).
- `src/clip/vision_wgsl.ts`: pure codegen (zero imports), one emitter per
  step kind, every shape/offset baked as literals — no uniforms, no runtime
  structure. Pointwise = tiled matmul (32px × 32cout workgroup tiles, x and W
  staged through shared memory). Spatial conv = one channel per workgroup
  (weights staged once), 4 pixels per thread, fully-unrolled register taps
  (dynamic-index private arrays spill to scratch on Metal), interior fast
  path + checked border path. Attention = pointwise(qkv) → per-head
  softmax core (K/V staged, score row in registers) → pointwise(proj, fused
  residual+layer-scale).
- `src/clip/vision.ts`: buffers + pipelines + one-compute-pass encoding.
  Device-agnostic (bun-webgpu and browser share the code path).

## Gotchas (hard-won, do not re-learn)

1. **Pin the input slot in κ.** See above; the bug is invisible to a
   single-forward test and deterministic garbage on the second forward.
2. **Warm up before benching.** Metal JITs each pipeline lazily on first
   use; a 10-run bench reads ~10 ms where steady state is 6.6 ms.
3. **bun-webgpu quirks:** one `requestDevice()` per adapter per process;
   `getCompilationInfo` missing (use validation error scopes — see
   `makePipeline` pattern); `readFileSync(...).buffer` is exact-sized in Bun.
4. **Per-step refs are the debugging superpower.** `dump_refs.py` re-exports
   every step's ONNX tensor as a graph output; the first FAIL in
   `fused_test.ts` names the exact broken kernel. Attention-internal steps
   have `ref: null` (our qkv/attn layouts differ from ONNX) and are covered
   by the block-output step.
5. **Directional-derivative ε must beat the fp32 noise floor.** The −cos loss
   is so insensitive that a unit perturbation moves L by only ~2e-7 at ε=1e-2 —
   right at fp32's rounding of an O(1) cosine — so FD is pure noise for small-
   derivative directions (the spec's ε=1e-2 gives only 3/8). An ε sweep shows FD
   converging monotonically to ⟨grad,v⟩ as ε grows (noise-limited, not
   truncation-limited over 0.03–0.2); ε=0.2 gives a clean 8/8. A measurement
   artifact, NOT a gradient error — the gate-1 units verify each formula at 1e-4.
6. **Workgroup memory ≤ 16 KB on this Dawn/Metal adapter.** `se_bwd` at c=1024
   wants 4 channel-sized scratch arrays (16.9 KB → pipeline validation error);
   `gap` and `dpre2` have disjoint lifetimes so they SHARE one (`tmp`) → 12.8 KB.
7. **Split-GELU keeps both refs.** In `--train` the Conv output (pre-GELU) and
   the GELU Mul_1 output are BOTH real ONNX tensors, so per-step bisection still
   works (`PLAN=plan_train.json` selects refs_train/ + weights_train.bin).

## Backward — dL/dpixels, weights FROZEN (`src/clip/vision_bwd_wgsl.ts`)

Hand-written WGSL backward (docs/clip_backward_spec.md) producing the loss
gradient **w.r.t. the input image only** — no dW anywhere. Feeds the splat
rasterizer backward so splats optimize toward a text prompt.

```bash
# 1. κ TRAIN plan: no slot reuse (activations saved for backward), GELU split
#    into its own step, transposed pointwise weights (wOffT) packed, backward:[…].
uv run --with onnx --with numpy python tools/clip/compile_plan.py --train
#    → plan_train.json (129 fwd steps, 152 backward entries) + weights_train.bin (86 MB)

# 2. train-plan per-step goldens (the split-GELU steps have real ONNX refs)
PLAN=plan_train.json uv run --with onnx --with numpy --with onnxruntime \
  python tools/clip/dump_refs.py                            # → refs_train/

# 3a. gate 3 — train FORWARD still matches ORT per step (cosine 1.000000)
PLAN=plan_train.json bun tools/clip/fused_test.ts
# 3b. gates 1+2 — per-kernel float64 refs + end-to-end directional derivative
bun tools/clip/bwd_test.ts                                   # GATE=1 → units only
```

Design (one kernel per backward kind, same pure-codegen discipline):
- `pw_bwd` REUSES the forward tiled matmul (`pointwiseTiledMain`, exported from
  vision_wgsl.ts) over the transposed `[Cout][Cin]` weights; a fused
  residual+layer-scale conv folds γ into `wOffT` at compile time and κ emits a
  tiny `residual_bwd` for the skip edge.
- `spatial_bwd` — gather mirror of the forward spatial conv, one thread per
  INPUT pixel, stride∈{1,2} baked. The stem's (cin=3) spatial_bwd IS dL/dpixels.
- `se_bwd` / `attn_core_bwd` recompute forward internals from the saved input
  (SE gap/gate; attention softmax from saved qkv, probs never stored), one
  workgroup each. `gelu_bwd` / `head_bwd` / `loss_bwd` are the obvious grads.
- Grad slots mirror activation slots; a tensor read by two forward edges gets
  `accumulate:true` on the later backward writer (κ orders writers — no global
  zero-fill, first writer overwrites).
- `VisionTrainer` (src/clip/vision.ts) owns activation+grad slots + a per-prompt
  text buffer, encodes forward + loss head + backward as ONE compute pass;
  dL/dpixels lands in `inputGradBuffer`.

Results (Apple metal-3, shared GPU ±30%, read trends): **gate 1** 16/16 units
(GPU vs float64 < 3e-7, analytic vs central-FD < 1.4e-5); **gate 2** 8/8
directional under rel 2e-2 (best 7e-4); **gate 3** 129/129 forward steps
relL∞ < 2e-3, cosine 1.000000; **bench** forward ~6–8 ms, **forward+backward
~20–40 ms (~2.7× forward)** — in the spec's 2–3× band. The gather `spatial_bwd`
kernels are correctness-first loops (an obvious perf follow-up: tile them).

## Text encoder (`tools/clip/text_onnx.mjs`)

The prompt side of the cosine loss. It runs ONCE per prompt change (not per
frame), so no hand-written kernel is needed — a stock runtime is fine:

```bash
node tools/clip/text_onnx.mjs ["a prompt"]      # default: "a photo of a cat"
```

`prompt → CLIP tokenizer (77-tok) → int64 input_ids → text_model_fp16.onnx →
text_embeds[512]`, all on ORT-CPU. Files auto-download (gitignored
`models/mobileclip_s0/`) from `Xenova/mobileclip_s0`: `tokenizer.json`,
`tokenizer_config.json`, `onnx/text_model_fp16.onnx` (85 MB fp16 weights, but
**int64 input / float32 output** — onnxruntime-node handles the fp16 interior).
Tokenizer is `@huggingface/transformers` `AutoTokenizer` pinned to LOCAL files
(`env.allowRemoteModels=false`); output embeddings written to
`fixtures/text_embeds_test.json` (`{prompt: [512 floats]}`).

**Verified** (`node tools/clip/text_onnx.mjs`):
- Tokenizer, empirically: BOS=49406, EOT=49407, `model_max_length=77`, pads to
  77 with token id **0** after the EOT (`tokenizer_config` `pad_token="!"` =
  vocab id 0, i.e. the CLIP "pad with zeros" convention). `"a photo of a cat"`
  → `[49406, 320, 1125, 539, 320, 2368, 49407, 0…]` (7 real + 70 pad).
- Embeddings are **not** L2-normalized (L2 ≈ 11 for the cat prompt).
- **Semantic sanity PASSES**: text↔text cosine `cat–dog = 0.9214` ≫
  `cat–diagram = 0.6483` (`dog–diagram = 0.6437`). Against the golden
  random-noise image the three text cosines are all low/similar
  (0.18–0.21) — expected for noise.

**Gotchas**
1. The fp16 export only loads at `graphOptimizationLevel:"basic"` — `"all"`
   and `"disabled"` both trip an ORT `SimplifiedLayerNormFusion` bug
   (`InsertedPrecisionFreeCast … does not exist`). fp16 was NOT awkward
   otherwise (int64 in / f32 out), so the `text_model_quantized.onnx` fallback
   was not needed.
2. Same pooled-Buffer trap as `ort_web_bench.mjs`: the golden image fixture is
   <4 KB, so `readFileSync(...).buffer` is the shared 8 KB pool — view it via
   `(buf.buffer, buf.byteOffset, buf.length/4)` or Float32Array reads garbage.

## Next (integration — HANDOFF §CLIP)

- Canvas → NCHW input kernel (particle renderer texture → slot 0; the /255
  is the whole preprocessing story).
- Text embedding: DONE (`tools/clip/text_onnx.mjs` above) — run per prompt
  change; feed it to `VisionTrainer.writeText` (the −cos loss head is baked in).
- Backward: DONE (see the Backward section above). `VisionTrainer.inputGradBuffer`
  holds dL/dpixels after one `run({backward:true})`; wire it into the force-field
  trainer (train_wgsl.ts) / splat rasterizer as a guidance term.

## Batch Experiments

The first isolated batch harness lives in `src/clip/vision_batch.ts` and
`tools/clip/batch_bench.ts`. It shares weights/pipelines across independent
image lanes and compares separate, lane-major, and step-major schedules:

```bash
BATCH=3 MODE=backward bun tools/clip/batch_bench.ts
BATCH=9 MODE=forward RUNS=10 WARMUP=3 bun tools/clip/batch_bench.ts
```

Current result: this replicated-activation batcher lowers CPU encode/submit
cost, but does not improve GPU wall time. See
`docs/CLIP_BATCHING_NOTES.md`; true sublinear batching needs a real batch
dimension in the generated WGSL kernels.

The first true batch-major forward fork lives in `src/clip/vision_batch_wgsl.ts`
and `BatchMajorVisionEncoder`. It verifies each lane against the original
single-image encoder and benchmarks `workgroups.z = batch`:

```bash
BATCH=9 RUNS=3 WARMUP=5 bun tools/clip/batch_major_forward_bench.ts
PLAN=plan_train.json BATCH=9 RUNS=3 WARMUP=5 bun tools/clip/batch_major_forward_bench.ts
```

Forward parity is exact for tested B=2/3/9.

The batch-major train fork verifies the optimizer-relevant gradient path:

```bash
BATCH=3 RUNS=2 WARMUP=3 bun tools/clip/batch_major_train_bench.ts
BATCH=9 RUNS=2 WARMUP=2 bun tools/clip/batch_major_train_bench.ts
TRIALS=2 CONFIGS='base=;early=8,10;candidates=8,10,111,115' bun tools/clip/batch_major_train_matrix.ts
TRIALS=2 CONFIGS='base=;stem=stem' bun tools/clip/batch_major_train_matrix.ts
TRIALS=2 CONFIGS='stem=stem;gelu=stem,gelu' bun tools/clip/batch_major_train_matrix.ts
TRIALS=2 CONFIGS='default=stem,gelu;gelubwd=stem,gelu,gelubwd' bun tools/clip/batch_major_train_matrix.ts
```

Gradient parity is exact for tested B=2/3/9. In the warmed train bench,
B=3 was about 2x faster than repeated single-image forward+backward, and B=9
was about 2.9x faster. The 3D optimizer now renders selected views into
batched CLIP input lanes and routes each lane's image gradient back through the
matching camera view using private raster forward state.

The 3D optimizer enables the stem spatial-backward specialization by default for
batch-major CLIP. Use `STEM_SPATIAL_BWD=0` in benches for the negative control.
It also enables pointwise + GELU forward fusion for batch-major CLIP; use
`FUSE_PW_GELU=0` in integrated 3D benches for that negative control.
GELU-backward into pointwise-backward fusion is available as a gated ablation
with `FUSE_GELU_BWD_PW=1`, but is not currently enabled by default.

The shared-weight pointwise microbench tests whether multiple image lanes can
reuse one staged W tile inside the same workgroup:

```bash
BATCH=2 STEP_INDEX=57 RUNS=40 WARMUP=10 bun tools/clip/pointwise_batch_bench.ts
BATCH=3 STEP_INDEX=8 RUNS=40 WARMUP=10 bun tools/clip/pointwise_batch_bench.ts
BATCHES=2,3 STEPS=8,10,57,59,111,113,115,117 TRIALS=2 bun tools/clip/pointwise_batch_matrix.ts
```

It verifies exactly, but results are shape-specific rather than universally
faster. Use it selectively after profiling full CLIP pointwise steps.

The dispatch profiler gives an isolated warmed ranking of generated WGSL
dispatches:

```bash
MODE=train BATCH=1 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
CSV=1 MODE=train BATCH=3 bun tools/clip/dispatch_profile.ts > /tmp/clip_b3.csv
BATCHES=1,3 TRIALS=2 RUNS=3 WARMUP=1 bun tools/clip/spatial_bwd_profile_matrix.ts
```

Treat it as a kernel-priority tool, not exact full-chain GPU timestamp
attribution. First profiles put `pw`, `pw_bwd`, `spatial_bwd`, and `conv` at
the top; attention backward is not first-wave work.

## prompt→splats browser page (Task #7 phase 2)

The end-to-end demo: type a prompt and a field of 2D Gaussian splats
live-optimizes on a canvas to match it (the optimizer defaults to the LEGIBLE
regime — ~12K large opaque splats, `LEGIBLE_LRS` — so a recognizable subject
appears in ~20 steps; the page passes NO `G`/LR overrides). Everything runs on
ONE `GPUDevice` the page creates
(`navigator.gpu`, no tfjs): the canvas context, the `SplatOptimizer`
(`src/splat/optimize.ts`), and a storage-buffer blit shader all share it. Text
is encoded ONCE per prompt by transformers.js (loaded from the jsDelivr CDN),
off the shared GPU (`device:"wasm"`, `dtype:"fp32"` → the fp32
`onnx/text_model.onnx`, dodging the fp16 layernorm-fusion gotcha above).

Files: `src/splat.html`, `src/splat_page.ts` (the wrap — DOM + WebGPU + blit +
rAF loop), `tools/splat/serve.mjs` (static server over the repo root),
`tools/splat/page_smoke.mjs` (puppeteer acceptance gate). `package.json`
`"source"` is now a 2-entry array so parcel builds both this and the existing
particle-art page.

**Run it**

```bash
# 1. vision train-weights (once; gitignored under models/mobileclip_s0/):
uv run --with onnx --with numpy python tools/clip/compile_plan.py --train
#    → plan_train.json (90 KB) + weights_train.bin (82 MB)

# 2. build the page (relative asset URLs so it serves under /dist/):
npx parcel build --no-scope-hoist --public-url ./ src/splat.html
#    (or `npm run build` to build BOTH pages)

# 3. serve the repo root — the page, the 82 MB weights and everything else are
#    then same-origin off ONE server (the text model + tokenizer + transformers.js
#    come from the HF hub / jsDelivr CDN at runtime, cached by the browser):
node tools/splat/serve.mjs                       # → http://localhost:8799

# 4. open http://localhost:8799/dist/splat.html , type a prompt, click Optimize.
```

**Why the vision weights are fetched, not bundled** — `models/` is gitignored
and outside `src/`, so parcel neither serves nor bundles it (82 MB through
parcel is a non-starter). The page `fetch()`es `/models/mobileclip_s0/{plan_train.json,
weights_train.bin}` off the static server. Only the vision weights need local
serving.

**Acceptance gate** — `node tools/splat/page_smoke.mjs` (after steps 1–2). It
serves the repo root, launches real headless-Metal Chrome (same flags as
`tools/qa_browser.mjs` + rAF-throttle-off so the 150 steps run in seconds),
types "a photo of a cat", waits for the text model + ~150 optimize steps, then
asserts (a) the live cosine ROSE from its initial value and (b) the canvas is
NON-BLANK (luminance variance of the decoded screenshot above a floor). Verified
green on Apple GPU (LEGIBLE regime): **cos 0.150 → 0.464 (Δ +0.313)**, canvas
variance 1479 ≫ 2, screenshot shows a clear central cat. First run downloads the
161 MB fp32 text model (cached in a persistent Chrome profile —
`/tmp/splat_smoke_chrome_profile`); with the model cached the whole gate is ~40 s
(most of it the ORT session init).

**Browser gotchas** (only surface at runtime):
1. The jsDelivr **`/+esm`** endpoint (not the raw `dist/transformers.web.js`) is
   required: the raw web build leaves `onnxruntime-web/webgpu` a bare specifier
   the browser can't resolve; `/+esm` rewrites deps to CDN URLs. The specifier is
   hidden behind `new Function("u","return import(u)")` so parcel emits a genuine
   native dynamic import instead of a bundle helper.
2. A WebGPU canvas's drawing buffer is **not preserved**: `canvas.drawImage`/
   `getImageData` into a 2D canvas in a later task reads all-zero even when the
   display is correct. The gate measures the **compositor screenshot** (decoded
   with `sharp`), not a canvas-to-canvas copy.
