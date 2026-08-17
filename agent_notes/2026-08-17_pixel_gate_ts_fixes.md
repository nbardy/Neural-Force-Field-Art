# 2026-08-17 — Pixel-gate silent fallbacks, two TS narrowing errors, editor type deps

Worktree: `.claude/worktrees/agent-a8dcffb8d594207ce`, branched from `main` @ `c342cb7`.
NOT committed. Source of the assignment: the "Two latent bugs found in the pixel
gate" section of `agent_notes/2026-08-17_120215_KST_fused_kernel_audit.md`.

---

## 1. What the audit claimed vs what the code actually does

The audit's line numbers are from an earlier revision (~190 lines of drift); the
mechanisms are otherwise exact.

| Audit claim | Verdict | Where it really is (pre-fix) |
|---|---|---|
| `if (advTrainerB) … else if (pixelDiscTrainer)` drops the pixel critic's extGradsBuf | **Confirmed [V]** | `src/main.ts:3702-3704` |
| the `extGradBuffers.length > 2` guard is unreachable | **Confirmed [V]** | `src/main.ts:3705`; max pushes = 1 (A) + 1 (B **xor** pixel) = 2 |
| `?train=tfjs` silently disables the pixel critic | **Confirmed, and worse than described [V]** | see §1.1 |
| no shipped piece sets both game + critic | **Confirmed [V]** | 4 Pixel pieces declare no `adversary`; the Agree+Disagree piece declares no `pixelDisc` (now asserted in a test) |

Two corrections to the audit:

- **The critic's `!wantTfjsTrainer` clause is redundant, not causal.** The
  construction site sits *inside* the fused-trainer block, which already
  requires `!wantTfjsTrainer` **and** `cfg.fieldLoss !== undefined` **and**
  `(advRt.tag === "off" || fusedAdvOk)`. So `?train=tfjs` kills the critic via
  the OUTER gate, and two further conditions the audit did not list (no
  `fieldLoss`; adversary routed to tfjs) silence it the same silent way.
- **Wiring the third buffer is not trivial.** `FusedTrainer` hard-caps at two
  (`src/render/webgpu/train.ts:185`) and the codegen refuses `extGradCount ∉
  [0,2]` (`src/render/webgpu/train_wgsl.ts:1238`), where each buffer is its own
  read-only binding (`:1371-1382`). A third claimant needs a new binding in
  `trainPassBShader` plus its own oracle gate — so this was fixed as a **loud
  typed error**, per the assignment's second option.

### 1.1 `?train=tfjs` on a Pixel piece is FATAL, not merely weaker — measured

Baseline (HEAD, `dist_base` build), headless Chrome on the real Metal adapter,
`?train=tfjs` then switch to `Pixel · VecField`:

```
[log] [advect] fused kernel: helmholtz, 3464 weight floats, …
[log] starting: Pixel · VecField (webgpu)
[pageerror] Cannot find a connection between any variable and the result of the
            loss function y=f(x). …
```

No `[pixel-disc]` line at all — the critic vanishes, the piece falls to
`computeLoss: helmholtzChaosLoss(ZERO_FIELD_LOSS)`, that loss is a constant with
no tape connection, and `optimizer.minimize` throws at frame 1. The user's only
signal was that opaque tfjs message. **Verified pre-existing**: reproduced on a
build of stashed HEAD sources (`dist_base`, port 8812) and on the fixed build
(port 8811) — identical throw, the fixed build just prints the explanation first.

---

## 2. Changes

### 2a. `src/render/webgpu/pixel_disc_wgsl.ts` — one source of truth for "is this field fusable"

Added `PixelDiscFusion = {tag:"ok"} | {tag:"unsupported";reason}` and
`classifyPixelDiscFusion(field)`; `validatePixelDiscFusion` now dispatches on it
(same throws, same messages modulo wording). The host needs the same answer
*without* throwing, plus a reason; two hand-copied ladders is how the constructor
and the gate drift apart.

### 2b. `src/main.ts` — pixel critic routing is now one typed decision

- `PixelCriticSpec` extracted from the inline `ArtPieceConfig["pixelDisc"]` type.
- `PixelCriticPlan = {absent} | {fused, spec} | {dropped, reason}` and
  `resolvePixelCritic(gates)` — exported, pure, GPU-free, so it is testable.
  - **Agree+Disagree game + `pixelDisc` THROWS** at piece-resolution time,
    before any trainer is constructed and before `[pixel-disc] FUSED` is ever
    logged. It throws on the *declaration*, independent of which trainer would
    have run: a piece carrying both is broken regardless.
  - Every other silencing gate returns a **named reason**: no field, `?train=tfjs`
    (naming the resulting tfjs throw), no `fieldLoss`, adversary-on-tfjs, plus
    the three field-shape reasons from `classifyPixelDiscFusion` (class-aware,
    hashgrid, non-two-head — the hashgrid one is reachable at runtime via the
    dock's arch preset, which is why it warns rather than throws).
  - The `fused` variant **carries the approved spec**, so the construction site
    performs no second check on `cfg.pixelDisc`.
- The call site sits with the rest of the trainer gate (just after the adversary
  routing decision) and `console.warn`s the `dropped` reason. A pointer comment
  next to the two existing `[train] ?train=tfjs ignored …` warns names the third
  consequence, so the ladder stays in one place instead of being duplicated.
- The construction `if` is now `pixelCritic.tag === "fused"`, followed by a
  **drift tripwire**: `fused` with no constructed trainer throws at startup.
- `extGradBuffers` is built from an explicit `extGradClaims` list (adversary,
  adversary lane B, pixel critic) — the `else` is gone, the count guard is
  reachable, and its message names the claimants.

### 2c. `src/main.ts` — the two real TS narrowing errors

Confirmed with `tsc --noEmit --strict` at exactly the reported lines:

```
src/main.ts(757,10): TS2339 Property 'kind' does not exist on type 'AdversarySpec'.
src/main.ts(2956,45): TS2339 Property 'weight' does not exist on type 'AdversaryRuntime'.
```

- **757** is `base.kind.tag` inside `resolveAdversary` (the variable is `base`,
  not `advSpec`). `from` is the already-narrowed piece spec and is non-null on
  that path, but the invariant spans statements. Fixed by reading `from.kind`
  behind an explicit `if (from === null) throw` — defaulting to `wta` would
  invent an adversary for a piece that declared none, i.e. the same bug class.
- **2956** is `advRt.weight` inside the per-frame `encodeAdversaryBranch`
  closure; TS drops narrowing of a mutable outer binding inside a closure. Fixed
  with `const advOn = advRt` inside the already-narrowed `if`, matching the
  file's own `rtSnap` precedent. Same object reference, so the dock's live
  weight slider is still read fresh each frame — no behavioural change.

No `as any`, no `!` assertions. `src/main.ts` now typechecks clean under
`--strict`.

### 2d. `tools/adversary_wire_test.ts` — §8 PIXEL CRITIC GATE

Nine checks, no mocks, on the real exported resolver:
the supported piece still resolves `fused` (regression guard for the four shipped
Pixel pieces); the plan carries the piece's own spec object (identity); the
Agree+Disagree combination throws, and its message names the game and the "0% of
the update" consequence; it throws regardless of trainer path; all six silencing
gates return a non-empty reason; `?train=tfjs` names itself; a piece with no
critic resolves `absent` (so the warn channel stays meaningful); and no gallery
piece declares the impossible pair.

### 2e. Type declarations (dev-only)

`npm install --save-dev @types/react@^18 @types/react-dom@^18 @types/node@^24`.
React/React-DOM are `18.2.0` in `package.json` and in `node_modules`, so the
types are pinned to the 18 line (a bare `npm i -D @types/react` installs v19).
`@types/node` pinned to the 24 line to match `node v24.3.0` on this box.
`npm` also re-synced `yarn.lock` (npm ≥ 7 updates an existing one) — that diff is
lockfile-only and includes dropping the already-removed `onnxruntime-web`.

---

## 3. Gate results

| Gate | Result |
|---|---|
| `npm run build` | clean, `✨ Built in 3.27s` |
| **@types install alone changes no bundle** | `dist/*.js|css|html` sha256 set **byte-identical** before vs after the install (build → install → rebuild → `diff` of hashes) |
| final bundle delta | only `index.*.js` (+ the `index.html` that references it). `index.387f0c14.css`, `splat.88455761.js`, `splat.html`, `splat3d.cdf3e425.js`, `splat3d.7ead7db1.js`, `splat3d.html` all unchanged |
| boot, default piece, real Metal adapter, headless | `[adversary] FUSED wta encoding=pair-rotation-scale-adjusted k=4 …`, `[train] fused trainer active (2 dispatches/step, …)`, 59.9 FPS, **0 page errors** |
| `Pixel · Inpaint` / `Pixel · VecField` after switch | `[pixel-disc] FUSED kind=… weight=0.04 …`, 0 errors — no regression |
| `Adversary · Agree + Disagree RGB` after switch | `[adversary] FUSED Agree+Disagree …`, 0 errors |
| **1a loud path, end-to-end** | temporarily added `pixelDisc` to the Agree+Disagree piece, rebuilt, switched to it in the browser: the throw fires as a page error **before** any `[pixel-disc] FUSED` log and before any trainer is built. Temporary edit reverted; verified absent from the diff |
| `bun tools/adversary_wire_test.ts` | ALL ADVERSARY WIRING CHECKS PASS (incl. new §8) |
| `bun tools/pixel_disc_test.ts` | ALL PASS (incl. §3 GPU extGrads — exercises the refactored `validatePixelDiscFusion`) |
| `bun tools/integration_test.ts`, `bun tools/drive_controls_test.ts` | ALL PASS |
| `tsc --noEmit --strict` on `src/main.ts` | 0 errors (was 2) |

Browser harness: `tools/smoke.mjs` cannot be used here — it forces
`forceFallbackAdapter: true` and there is no SwiftShader-WebGPU adapter on this
box. A scratch harness with
`["--no-sandbox","--enable-unsafe-webgpu","--enable-webgpu-developer-features","--ignore-gpu-blocklist","--use-angle=metal"]`
gets a real adapter (`PROBE {"adapter":"yes"}`) and renders at 59.9 FPS headless.
**Worth folding into `tools/smoke.mjs` as a `--real-adapter` mode.**

---

## 4. Left alone deliberately (findings, not fixes)

1. **`?train=tfjs` on a Pixel piece still ends in the tfjs throw.** The
   assignment asked for a warn, not a behaviour change, so `?train=tfjs` is
   still honoured. The consistent fix is to *ignore* it for `cfg.pixelDisc`
   pieces exactly as the class-aware and Agree+Disagree cases do
   (`wantTfjsTrainer = false` + warn), since the tfjs path has no critic AND no
   usable loss on those pieces — it is a dead page, not an A/B comparison.
2. **`// TEMP INSTRUMENTATION — remove before commit`** is live at
   `src/main.ts:2993` (HEAD, not from this work). Its
   `[loop] tick bailed without rescheduling [object Object]` warn fires on
   **every** piece switch (the outgoing loop's last tick) — it is noise that
   trains readers to ignore warnings, which directly undercuts the fixes above.
3. **11 pre-existing `tsc --strict` errors outside `src/main.ts`**, all
   mechanical: 8 × `timestampWrites` (`@webgpu/types` in the tree predates the
   `GPUComputePassTimestampWrite` → object-form change) in `advect.ts`,
   `adversary_train.ts`, `pixel_disc_train.ts`, `splat.ts`, `train.ts`;
   `adversary_wgsl.ts:870` `Type 'string' is not assignable to 'never'`;
   `core/gan/adversary.ts:644` tfjs `customGrad` signature. None affect the
   parcel build (parcel transpiles, it does not typecheck).
4. **`PixelDiscTrainer.lastStats` is still dead** (audit §1). Untouched.
5. The repo still has no `tsconfig.json`; the typecheck above was run with
   explicit flags from a scratch TypeScript 5.6. Adding a `tsconfig.json` would
   make the editor and any future CI agree on the same 13 → 11 errors.
