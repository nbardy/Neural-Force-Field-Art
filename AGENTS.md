# AGENTS.md

## Agent notes

For any non-trivial investigation, optimization, multi-agent task, or work that
may span sessions, create or update a timestamped note under `agent_notes/`:

```text
agent_notes/YYYY-MM-DD_HHMMSS_TZ_<topic>.md
```

The note is the durable handoff. Record the goal, questions raised, files and
commands inspected, measured results, unresolved concerns, and next actions.
Distinguish verified facts from hypotheses and proposals. Do not rely on chat,
workflow notifications, or an agent's final message as the only record of
material work; inspect the worktree and keep the note current.

## Cursor Cloud specific instructions

### Project overview

Neural Force Field Art — a single-page React 18 app. Tiny neural force fields
move particle clouds. The production field trainer, adversarial trainer,
particle integrator, diagnostics, and renderers are fused WebGPU paths; TF.js is
retained as the independent reference/oracle and fallback trainer. Bundled with
Parcel.

### Dev server

```bash
yarn start          # runs `parcel src/index.html` on port 1234
yarn build          # production build to dist/
```

### Rendering & backend (current — WebGPU-only)

- `main.ts` forces `tf.setBackend("webgpu")`. Particle state lives in
  `AdvectKernel`-owned `GPUBuffer`s. `src/render/webgpu/points.ts` and
  `src/render/webgpu/splat.ts` bind those buffers directly, so the production
  particle path is zero-copy and has no state readback. `FusedTrainer` and
  `AdversaryTrainer` update disjoint packed weight/Adam segments and feed the
  same field buffers. The renderer shares tfjs's `GPUDevice`
  (`tf.backend().device`). If WebGPU is unavailable it shows a "This needs
  WebGPU" notice — there is **no** Canvas2D/WebGL fallback, by design.
- **Three WebGPU gotchas already handled in `main.ts` — keep them:**
  1. `import "@tensorflow/tfjs-backend-webgpu"` — the tfjs union package only registers cpu+webgl; without this `setBackend("webgpu")` throws _"Backend name 'webgpu' not found in registry"_.
  2. A `GPUAdapter.prototype.requestAdapterInfo` shim — tfjs 4.10 calls that removed API; current Chrome exposes `adapter.info`.
  3. **All tensor/model creation is deferred until after `await tf.ready()`** — webgpu init is async; building tensors earlier throws _"backend not yet initialized"_.
- `src/renderers.ts` (Canvas2D) is a dead implementation kept as a revert path;
  production imports only its `RendererType` type. `src/render/gpuPoints.ts`
  (WebGL2) is also a non-production revert path.

### Verifying a WebGPU page headless: `tools/smoke.mjs`

Most headless browsers can't render WebGPU, so an agent can't "see" the page. `tools/smoke.mjs` drives any URL in headless Chrome (Puppeteer) with WebGPU flags + a forced SwiftShader **software fallback adapter**, captures ALL console output + page errors, probes `navigator.gpu` / the tfjs backend / the on-screen HUD, and writes a screenshot — printing the screenshot PATH so you can Read it.

```bash
node tools/smoke.mjs [url] [waitMs] [outDir]
node tools/smoke.mjs http://localhost:8798/index.html 8000 /tmp
node tools/smoke.mjs https://nbardy.github.io/Neural-Force-Field-Art/
```

Output: `SCREENSHOT <path>`, `PROBE {webgpu,adapter,hud,warning}`, then the full console log. **Caveat:** whether a *software* WebGPU adapter is available depends on the box — some headless Chromium builds report `adapter: null` (no SwiftShader-WebGPU), in which case the app correctly shows the WebGPU warning and only the non-render paths are verifiable. On a machine with a real (or SwiftShader) WebGPU adapter it renders and the screenshot shows particles.

**On an Apple box `smoke.mjs` does not work and is not the tool you want.** It
forces a SOFTWARE fallback adapter that does not exist there, so the page
correctly shows the "needs WebGPU" notice and nothing renders — a run that
looks like a regression and is not. Use the REAL-adapter flags that
`tools/health_audit.mjs`, `tools/soak_adversary.mjs` and
`tools/pressure_live_probe.mjs` already share (`--enable-unsafe-webgpu
--use-angle=metal --ignore-gpu-blocklist`), or just run the audit below — it
drives the gallery, records the metrics, and screenshots with `HEALTH_SHOTS=1`.

### Tests: `bun tools/<name>_test.ts`

There is no aggregate runner, no `yarn test`, no watch mode — **31 `*_test.ts`
suites in `tools/`, run one file at a time.** Each prints `ok`/`FAIL` lines and
exits nonzero on failure.

```bash
bun tools/field_probe_test.ts
```

**21 of the 31 open a real WebGPU device and must run SEQUENTIALLY**, with
nothing else on the GPU — parallel runs flake for reasons that have nothing to
do with your change. Tell them apart mechanically rather than from a list here
(a list goes stale):

```bash
rg -l bun-webgpu tools/*_test.ts   # GPU: serialize these
```

Glob the test files, not the directory: `tools/` also holds probes and a
`tools/splat/` subtree that open a device without being suites.

The other 10 are pure CPU (`ad_test`, `ad_jvp_test`, `ad_train_test`,
`adversary_test`, `adversary_objectives_test`, `adversary_strict_test`,
`adversary_wire_test`, `cover_oracle_test`, `drive_controls_test`,
`url_guard_test`) and are safe to run together.

**`adversary_strict_test` is RED and has been for a while** — bisected to
eb211d1 / 7f4dba0 / 7a8d12d, so it predates any of the dock work above. It
fails the adjusted-observer invariants (`active target value is exactly
invariant under positive signal scaling`, `active target is an exact unit
direction`, `public predictions are direction-normalized`). That is the strict
suite for the observer the Pair pieces use, so treat a red run as EXPECTED
noise for now and diff against these three names before blaming your change —
but it is unowned, and it should not stay that way.

Suites that gate a specific invariant are named at the invariant: see
`tools/family_grid_test.ts` under Particle families, `tools/field_probe_test.ts`
under Health metrics, and the probes under Adversary numerical stability.

### Health metrics: `window.__nffHealth`

Every piece publishes a structured **~1 Hz snapshot of EXACT floats** — never
formatted text. Schema, and what each number detects, is in `src/health.ts`;
the measurement is `src/render/webgpu/field_probe.ts`, which builds its own
encoder and binds the weights READ-ONLY, so it cannot perturb the artwork
(`field_probe_test.ts` §4 enforces exactly that, including a bit-identical
weight check across 5 samples).

Three blocks: `field` (present on **every** piece), `adv` (`null` unless the
piece has an adversary), `pixel` (`null` unless a pixel critic runs).

Read the thresholds from `src/health.ts` rather than restating them. Three
contracts that are easy to get wrong:

- **`null` always means UNMEASURED, never 0.** An unmeasured R₁ reported as 0
  reads as "perfectly isotropic" — the most flattering lie this instrument
  could tell about a collapsed field.
- **Direction collapse is `r1`, and `field.r1` ≠ `adv.r1`.** `field.r1` is the
  32² GRID measurement and exists on every piece; `adv.r1` is the training
  BATCH twin and is `null` unless anti-collapse pressure is compiled. Same
  statistic, same τ = 0.05, two sample populations — never substitute one for
  the other, and note that the gap between them is itself a reading (the cloud
  has bunched into part of the domain). Measured: pressure ON → grid R₁ 0.003;
  the same piece with `?advPolar=0&advNematic=0` → **0.52**, DC/AC **72**.
- **Never read `r1` without `r2`.** R₁ alone is escapable: a ±F₀
  counter-streaming field scores R₁ ≈ 0 and looks exactly as laminar. Measured
  R₂ = 0.81 on "Neural Field · Max Structure" while its R₁ sat at 0.10.
  **This is now GATED**, as `nematic-collapse` at `HEALTH_R2_MAX` (default 0.5),
  ranked below both laminar arms because polar order implies nematic order. It
  was prose here and nothing else for weeks, and the piece named in the sentence
  above was scoring PASS the whole time — re-measured 2026-08-25 at R₁ 0.034 /
  **R₂ 0.927**. A failure mode described in this file and not gated in
  `classify()` reads, to everyone downstream, as a check that is running.

**Never parse the HUD to get these.** Every previous headless gate regexed
numbers out of the on-screen text, and the 2026-08-17 soak flake
(`agent_notes/2026-08-17_soak_flake_attribution.md`) was the bill: a formatter
change or a `toExponential(2)` rounding across a threshold is indistinguishable
from the artwork changing.

**Recording them** — headless, real adapter, one typed verdict per piece:

```bash
node tools/health_audit.mjs --self-test
node tools/health_audit.mjs hashgrid,struct http://localhost:1234/index.html 60 2
```

`?arch=<preset>` selects the field architecture (keys in `src/core/field/arch.ts`
`ARCH`). Like `?advM`/`?advK` it is GLOBAL — it survives a gallery switch, which
is what lets a sweep vary architecture across pieces — and an unknown value
THROWS rather than falling back to the piece default. It is honoured only on
`archEditable` pieces; the snapshot's measured `arch` block is how you tell.

`--self-test` is pure (no GPU, no server) and gates the verdict logic itself. A
real run writes a per-piece time series plus `summary.json` to
`output/health-audit/<iso-timestamp>/` (gitignored) and **exits with the number
of unhealthy pieces**. Piece keys are the `PIECES` map at the top of that file;
`all`, `adversary` and `pixel` are group aliases; `HEALTH_SHOTS=1` adds
screenshots. **`all` means all 16 gallery pieces.** Until 2026-08-25 it covered
10 and said so in its own comment, so every green `all` run was a claim about
62% of the gallery — and both of the worst-collapsed fields were in the
uncovered 38%. Keep `PIECES` in sync with `GALLERY` in `src/main.ts`.

Importing this file is side-effect free (`INVOKED_DIRECTLY` guard) — that is
what lets `tools/health_sweep.ts` reuse `runPiece`/`aggregate`/`classify`.
Before the guard, `import()` launched a full GPU audit.
Every gate threshold is overridable by env (`HEALTH_R1_MAX`, `HEALTH_AC_DEAD`,
…) and every default is a measured number from `agent_notes/`, not a guess.

Design + the readings above:
`agent_notes/2026-08-19_032513_KST_grid_direction_order.md`.

#### Sweeps: the same instrument across a config MATRIX

`health_audit.mjs` measures ONE config — the right unit for "is the shipped
gallery healthy", the wrong one for "which objective/architecture actually
produces structure". That second question is `tools/health_sweep.ts` (driver)
plus `tools/health_report.ts` (reducer). They are separate because a matrix is
hours of serialized GPU and the reduction is milliseconds: the report can be
re-cut against a new gate without re-measuring anything.

```bash
bun tools/health_sweep.ts --example > sweep.json
bun tools/health_sweep.ts sweep.json http://localhost:1234/index.html
bun tools/health_report.ts output/health-sweep/<spec-name>
```

The spec is `{name, durationSec, sampleSec, axes}`. `axes.piece` takes the
audit's short piece keys; **every other axis name is a URL knob passed
verbatim** (`arch`, `advLoss`, `advM`, `advK`, `advPolar`, `advNematic`,
`advWeight`, `gLR`, `dLR`, …), so a knob added to the app is sweepable the day
it lands. Both files have `--self-test` (pure, no GPU, no server).

Three properties worth knowing before you trust a report:

- **It resumes.** The run directory is keyed by `spec.name`, not a timestamp,
  and a cell whose result file *parses* is skipped. Re-invoking the same spec
  continues an interrupted sweep; a truncated file is re-run rather than
  counted. Rename the spec to force a fresh run.
- **It proves the axes actually varied.** `?arch=` is honoured only on
  `archEditable` pieces and adversary knobs are inert without an adversary, so
  a sweep that trusted its own URLs would run one network N times and conclude
  "architecture makes no difference". Every cell records the MEASURED
  `ArchHealth` fingerprint (`src/health.ts`) and the report puts a **collision
  banner above every chart** when an axis produced one compiled network.
  Verified live: `arch=dualStd` → `raw/w2440`, `arch=dualFourier` →
  `fourier/w3464`.
- **Unmeasured is never plotted as 0.** Nonfinite samples break the trace into
  separate polylines, so a gap looks like a gap instead of a line drawn through
  values nobody measured.

Wall clock is the real constraint: the audit is serialized on purpose (parallel
runs measure GPU scheduler contention, not the artwork), so 4 objectives x 3
archs x 5 pieces x 300 s is over five hours. The driver prints the estimate
before it opens the browser.

### Build

- `yarn build` = `parcel build --no-scope-hoist`. **`--no-scope-hoist` is load-bearing:** default scope-hoisting crashes tfjs at runtime (`ReferenceError: $<hash>$exports is not defined`, blank page).
- Clear `.parcel-cache`/`dist` after switching branches if you hit `Expected content key … to exist`.

### Deploy: always use `tools/deploy.sh`

```
tools/deploy.sh "commit message"   # commits the working tree first
tools/deploy.sh                    # tree must already be clean
```

Build → commit `main` → push `main` → publish `gh-pages` → verify the live site. Run this instead of hand-rolling the steps; the two flags it bakes in are the ones a manual build gets wrong:

- **`--public-url ./`** — Pages serves this repo from `/Neural-Force-Field-Art/`, not the domain root. Without it parcel emits absolute `/index.<hash>.js` paths that 404 live *while the HTML still returns 200*, so a curl check passes and the page is blank. Hit for real on 2026-08-15; the script now fails the build if root-absolute asset paths appear.
- **`--no-scope-hoist`** — see Build above.

It also waits for the asynchronous Pages build and asserts the live `index.html` references the new content-hashed bundle, because a 200 returned before that build lands is the *previous* deploy.

### Caveats

- No linter, and no aggregate test runner — but there ARE 31 suites, run
  per-file with `bun`. See **Tests** above; do not conclude from this bullet
  that a change ships unverified.
- Both `yarn.lock` and `package-lock.json` may exist; the deploy path uses `npm` + `git push` to the `gh-pages` branch (site: https://nbardy.github.io/Neural-Force-Field-Art/).
- `tools/smoke.mjs` needs `puppeteer` (a devDependency; downloads a Chromium on install).
- **`yarn` may not be on PATH** even though every command here is spelled
  `yarn …`. `node_modules/.bin/parcel src/index.html --port 1234` is the direct
  equivalent of `yarn start`, and is what a `.claude/launch.json` should invoke
  (that file is gitignored, so recreate it per-machine).
- **The in-app browser pane cannot verify this app.** It hands the page a 0×0
  canvas, so the swapchain fails, the render path never runs, and
  `__nffHealth.field` stays `null` — indistinguishable from a dead field, and
  a false negative you will chase. Use a real headless Chrome with the
  real-adapter flags (`tools/health_audit.mjs`, `tools/dock_swap_probe.mjs`,
  `tools/soak_adversary.mjs`) instead.

#### Particle families (RGB Families piece)

A particle's family label is **never stored** — it is
`pcg(i ^ CLASS_SALT) % C`, derived identically in the advect kernel, both
trainers, the adversary and the renderer. Three copies of that derivation must
agree or the cloud is advected by one family's field and coloured as another,
silently.

**Two routes, one κ.** `FamilyRoute` (`advect_wgsl.ts`) is the only place the
routing is decided; every consumer dispatches on the tag and none re-derives it
from `classes` + `encoding.kind`:

- `onehot` — RAW encoding only. C one-hot channels on head 1's layer-0 input
  (the `Neural Field · Species` piece). **The fused adversary refuses this** —
  those extra rows have no counterpart in its field backward.
- `grid-plane` — HASHGRID only. `planes: C` stacks one feature table per family
  and every cell index carries `cls · gridSize²`. Fully supported by the fused
  adversary; the label rides the dEnc machinery the reward already uses.

Dropping the `cls · gridSize²` term is the failure mode to watch: it compiles,
runs, trains, and quietly collapses the three families onto one shared plane.
`tools/family_grid_test.ts` gates the offset directly (FD vs analytic, plane
isolation, per-plane training displacement) — run it after touching any grid
indexing. `planes: 1` regenerates the pre-family WGSL byte for byte, which is
what protects the shipped hashgrid pieces.

Per-family payoff telemetry requires **m == 1** (the point observer): only then
does a tuple have one unambiguous family. `familyInstrument` reports the typed
state `off`/`unmeasured` otherwise rather than inventing a bucketing rule.

Design + measurements:
`agent_notes/2026-08-19_family_conditioned_hashgrid_adversary.md`.

## Model selection: the TWO nets a relational game trains

"What model is this piece?" has two answers, and conflating them is the most
common way to misread a result.

- **Generator** — the neural force field itself, declared as a `FieldArch`
  (`src/core/field/arch.ts`): encoding (raw / fourier / hashgrid) × activation
  (selu / sin) × widths × heads (1 or 2) × α blend. Thirteen `ARCH` presets.
- **Predictor / discriminator** — K heads, each
  `du → hiddenUnits selu → featureDim selu → dy linear`. `du` and `dy` come from
  the OBSERVER (`fusedObjectiveDims`), not from the preset.

### Both are dock knobs, and both are compiled

A piece opts in with declarative `fieldArch` + `archEditable` + `archDock`
rather than `createField`; see `DUAL_ARCH_DOCK` in `main.ts` for the single
canonical explanation and the current membership. `?arch=<preset>` /
`?advHidden=<n>` / `?advFeature=<n>` are the URL equivalents, all GLOBAL (they
survive a gallery switch) and all THROWING on an out-of-range value rather than
falling back to the piece default.

`createField` is the hatch for pieces whose arch genuinely bakes semantics, and
exactly two need it: **Agree + Disagree RGB** (`fourierOctaves: 3`, which
`applyArchDockPreset` deliberately does not carry across a swap) and **RGB
Families · HashGrid** (`grid-plane` FamilyRoute — no dual preset carries
`classes`/`planes`). `applyArchDockPreset`'s preserve list is α / `semantic` /
`classes` and the reasoning for what is NOT on it is documented there.

**Cost of the hatch, when it is used where it is not needed:** `index.tsx`
gates the WHOLE model section on `piece.fieldArch`, so a `createField` piece
shows no arch info at all — not even the read-only summary. Those two pieces
still cannot tell you what they are running.

### EVERY prior adversary reading is a 32/16 reading

`AdversaryTrainer` and the tfjs `Adversary` have accepted
`hiddenUnits`/`featureDim` since the port and **no caller passed either until
2026-08-26**. The predictor is the one model in this project that had never
been varied. Win EMAs, payoff curves, R₁/R₂, the pole-exploit refutation, the
NaN probes — all of it is a property of a **32/16 adversary**, not of "the
adversary". Resolve the pair through `predictorArchOf` and pass it explicitly;
both trainers still carry their own `?? 32` / `?? 16`, which is two places for
the fused path and its own oracle to drift on a number neither declares.

The only predictor restriction in the fused codegen is activation: SELU, not
`sin`, because `emitBwdStore` needs pre-activation checkpoints `sin` does not
keep. **That refusal walks the PREDICTOR, not the field** — which is why
`dualSiren` (a SIREN *generator*) is accepted. Misreading
`validateAdversaryFusion` as "SIREN unsupported" is easy and wrong.

### Gating a dock that can recompile the artwork

`ARCH_DOCK_DUAL` happening to equal the set `validateAdversaryFusion` accepts
is a coincidence of two files. `adversary_wire_test.ts` makes it load-bearing
by CODEGEN over every combination the dock can produce — §8d for the pixel
critics, §8e for the relational adversary (7 pieces × 4 arch presets × 5
predictor widths = 140, ~12 s, pure CPU). Emitting the shaders is the gate, not
just building the layout: the refusals that bite live in the emitters.

> **Both gates must honour `arch.heads`.** §8d originally hardcoded a two-head
> `layoutField`, so a single-head preset added to the dual dock produced a
> helmholtz layout in the test and a `vector` layout in production — the gate
> stayed GREEN while the dock offered a field the critic refuses. It could not
> have caught the thing it exists to catch. Found 2026-08-26 by injecting
> `ARCH.siren` into `ARCH_DOCK_DUAL`; §8e failed and §8d did not.
>
> **That injection is how you verify either gate still works.** A codegen gate
> that cannot be made to fail is decoration.

### Proving a swap RUNS, not just compiles

```bash
node_modules/.bin/parcel src/index.html --port 1234   # separate shell
node tools/dock_swap_probe.mjs [url] [piece] [settleMs]
```

Drives the real dock, swaps arch + predictor mid-flight, and reads
`window.__nffHealth` (exact floats, never the HUD) on both sides. Measured on
"Adversary · Pair WTA K=4":

| | field arch | predictor | grid R₁ | grid R₂ | fps |
|---|---|---|---|---|---|
| before | standard [32×32] · 2 heads | 4 × [32, 16] | 0.038 | 0.0018 | 60.0 |
| after | hashgrid [32×32] · 2 heads | 4 × [64, 32] | 0.0038 | 0.000023 | 60.0 |

Both nets genuinely recompile — the advect log goes 2440 → 6664 weight floats
(f16/unrolled → f32/not) and the adversary log goes `predictor=32/16` →
`predictor=64/32`. Read R₁ **with** R₂, per Health metrics above.

Design + measurements:
`agent_notes/2026-08-26_015500_KST_adversary_dock_arch_and_predictor.md` and
`agent_notes/2026-08-25_122500_KST_pair_wta_arch_dock.md`.

## Adversary numerical stability

- Standalone adversary pieces use field/G Adam LR `0.001`, D LR `0.003`. Reward
  weight is **not** an honest generator-step knob under Adam (see
  `agent_notes/2026-07-29_010655_KST_adversary_release_completion.md`).
- `tools/adversary_stability_probe.ts` co-trains D→G on a shared field buffer
  **without advect** — useful, but it missed the Quad/Tri live blow-up.
- Particle-coupled repro: `TAG=quad-labelled N=60000 bun tools/quad_nan_probe.ts`
  (advect + particle-sourced D/G). Use `FINE_AFTER=<step>` to catch the first
  nonfinite **stage**. Live HUD soak: `node tools/soak_adversary.mjs quad6 …`.
- Policy in fused WGSL: ε inside every `sqrt`/norm radicand, activity floors on
  singular charts, SELU preact clamp (±80), and `isFiniteF` gates on residuals /
  Adam / extGrads. Guards are **inline in the fused shaders** — no host
  round-trip; the hot path remains one `encodeStep` encoder → one `queue.submit`.
- Detail handoff: `agent_notes/2026-08-05_231314_KST_quad_wta_nan.md`.
