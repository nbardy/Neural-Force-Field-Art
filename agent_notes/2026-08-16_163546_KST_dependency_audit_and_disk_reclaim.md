# Dependency audit and disk reclaim

**Date:** 2026-08-16 16:35 KST
**Trigger:** the machine hit 100% disk. `.parcel-cache` was found at **5.5 GB**
against a **2.9 MB** `src/` — roughly 1,900× the source — which prompted a check
of whether the bundler and the dependency set are carrying dead weight.

## Verified facts

### Why the cache is 5.5 GB (not a misconfiguration)

Parcel's cache is an LMDB store (`.parcel-cache/data.mdb`, 5.1 GB of the total)
that **grows monotonically and never compacts**. Stale entries from superseded
builds are never reclaimed. Timestamps inside the directory span 2026-07-08 to
2026-08-11 — five weeks of dev-server sessions accumulating with zero eviction.

Parcel exposes no cache-size cap and no eviction policy; the only knobs are
`--cache-dir` (relocate) and `--no-cache` (disable). `yarn build` already passes
`--no-cache`, so **the entire 5.5 GB came from `yarn start` dev sessions**.

The dependency set makes this worse than typical: tfjs (core + webgpu backend +
vis) is cached in transformed form, and re-cached on every version bump.

Existing guidance in AGENTS.md already says to clear `.parcel-cache`/`dist`
after switching branches (the `Expected content key … to exist` failure). This
note adds the disk-cost reason to do it on a schedule, not only on that error.

### Build output

**12 `dist*` directories totalling 3.58 GB**, `dist/` alone 2.97 GB containing
**199 `*.js.map` files at ~42 MB each**. Parcel writes content-hashed outputs
and nothing prunes superseded ones, so every rebuild leaves its predecessor
behind. `.gitignore` line `dist*/` already covers all variants — they were never
tracked; this is purely local disk.

### Dependency usage (measured by import, excluding `node_modules`/`dist*`)

| Package | Verdict | Evidence |
|---|---|---|
| `@tensorflow/tfjs` (+ backend-webgpu, vis) | **KEEP** | imported by **42 files** across `src/draw`, `src/agentSets`, `src/physics`, `src/utils`; AGENTS.md retains it as the independent reference/oracle and fallback trainer |
| `twgl.js` | KEEP | `src/draw/draw_webgl.ts`, `src/render/gpuPoints.ts`, `src/quickDraw/main.ts` |
| `react` / `react-dom` | KEEP | `src/index.tsx` (single entry) |
| `puppeteer` | KEEP | `tools/qa_handoff.mjs`, `tools/smoke.mjs` |
| `bun-webgpu` | KEEP | `tools/quad_nan_probe.ts`, `tools/border_modes_test.ts` |
| `onnxruntime-node` | KEEP | `tools/clip/text_onnx.mjs` (Node-side CLIP text encoder) |
| `@huggingface/transformers` | KEEP | `tools/splat3d/grid_quality.ts`, `tools/splat3d/aniso_quality.ts` |
| **`onnxruntime-web`** | **REMOVED** | zero imports anywhere. The browser gets ORT transitively from the **CDN** copy of transformers.js — see `src/splat_page.ts:194-203`, which loads `https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm` through a `Function`-constructor indirection specifically so the bundler leaves the specifier alone. The npm package was never in the bundle. |
| **`webgpu`** | **REMOVED** | zero imports. It is `"WebGPU for Node [WIP]"`, a runtime binding, not types. TypeScript types come from `@webgpu/types`, which is installed independently. |

Note the browser/tools split is already correct: everything used only by
`tools/` sits in `devDependencies`, and the `src/` runtime deps sit in
`dependencies`. No restructuring was warranted — only the two dead entries.

## Changes made

- `package.json`: removed `onnxruntime-web` (devDependencies) and `webgpu`
  (dependencies). No import sites existed, so the bundle is unaffected.
- `.gitignore`: removed a duplicated `node_modules` line.

**Lockfiles were NOT regenerated.** Both `bun.lock` and `package-lock.json`
exist in this repo (AGENTS.md acknowledges the dual-lockfile situation), and
re-locking is a separate decision about which package manager owns the repo.
Until that happens the lockfiles still list the two removed packages; they are
inert because nothing imports them.

## Not verified

A build was deliberately **not** used as the check. `node_modules` still
contains both removed packages, so a passing build would prove nothing about
whether `package.json` still needs them. The static import sweep is the sound
test, and it is what was run.

## Open items / proposals

1. **Bundler.** Parcel's non-compacting cache is structural, not tunable. Vite
   is the low-risk alternative: native multi-page HTML entries matching the
   three current entrypoints, and a dev cache (`node_modules/.vite`) measured in
   tens of MB that re-optimizes only on dependency change. Bun is tempting for
   toolchain unification (`bun-webgpu` is already here), but the tfjs +
   WASM-heavy dep set is exactly where a younger bundler costs time, and
   tfjs-under-Bun shims have already been needed in the headless path. Proposal:
   keep Bun for headless verification, evaluate Vite for the browser bundle.
   **`--no-scope-hoist` is load-bearing** (default scope-hoisting crashes tfjs
   with `ReferenceError: $<hash>$exports is not defined`) — any migration must
   confirm the replacement bundler does not scope-hoist tfjs into the same
   failure.
2. **Dual lockfiles** (`bun.lock` + `package-lock.json`) will drift. Pick one.
3. **`dist*` accumulation** deserves a prune step in `tools/deploy.sh`, which
   currently creates timestamped `dist_deploy_*` directories that are never
   collected.
