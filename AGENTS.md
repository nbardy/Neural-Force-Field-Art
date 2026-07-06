# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Neural Force Field Art — a single-page React 18 app. A tiny neural network (the "force field") is trained with TF.js autograd to move particles; the whole thing runs on the GPU via **WebGPU** (zero-copy tfjs `GPUBuffer` → WGSL renderer). Bundled with Parcel.

### Dev server

```bash
yarn start          # runs `parcel src/index.html` on port 1234
yarn build          # production build to dist/
```

### Rendering & backend (current — WebGPU-only)

- `main.ts` forces `tf.setBackend("webgpu")`. Rendering is `src/render/webgpu/points.ts` (+ the `microgpu.ts` helper): particle positions are read straight from tfjs tensors via `tensor.dataToGPU()` (a `GPUBuffer` on the webgpu backend) and drawn with an instanced WGSL round-dot shader — **true zero-copy, no readback**. The renderer shares tfjs's `GPUDevice` (`tf.backend().device`). If WebGPU is unavailable it shows a "This needs WebGPU" notice — there is **no** Canvas2D/WebGL fallback, by design.
- **Three WebGPU gotchas already handled in `main.ts` — keep them:**
  1. `import "@tensorflow/tfjs-backend-webgpu"` — the tfjs union package only registers cpu+webgl; without this `setBackend("webgpu")` throws _"Backend name 'webgpu' not found in registry"_.
  2. A `GPUAdapter.prototype.requestAdapterInfo` shim — tfjs 4.10 calls that removed API; current Chrome exposes `adapter.info`.
  3. **All tensor/model creation is deferred until after `await tf.ready()`** — webgpu init is async; building tensors earlier throws _"backend not yet initialized"_.
- `src/renderers.ts` (Canvas2D) and `src/render/gpuPoints.ts` (WebGL2) are **dead code** kept as a revert path; nothing imports them at runtime.

### Verifying a WebGPU page headless: `tools/smoke.mjs`

Most headless browsers can't render WebGPU, so an agent can't "see" the page. `tools/smoke.mjs` drives any URL in headless Chrome (Puppeteer) with WebGPU flags + a forced SwiftShader **software fallback adapter**, captures ALL console output + page errors, probes `navigator.gpu` / the tfjs backend / the on-screen HUD, and writes a screenshot — printing the screenshot PATH so you can Read it.

```bash
node tools/smoke.mjs [url] [waitMs] [outDir]
node tools/smoke.mjs http://localhost:8798/index.html 8000 /tmp
node tools/smoke.mjs https://nbardy.github.io/Neural-Force-Field-Art/
```

Output: `SCREENSHOT <path>`, `PROBE {webgpu,adapter,hud,warning}`, then the full console log. **Caveat:** whether a *software* WebGPU adapter is available depends on the box — some headless Chromium builds report `adapter: null` (no SwiftShader-WebGPU), in which case the app correctly shows the WebGPU warning and only the non-render paths are verifiable. On a machine with a real (or SwiftShader) WebGPU adapter it renders and the screenshot shows particles.

### Build

- `yarn build` = `parcel build --no-scope-hoist`. **`--no-scope-hoist` is load-bearing:** default scope-hoisting crashes tfjs at runtime (`ReferenceError: $<hash>$exports is not defined`, blank page).
- Clear `.parcel-cache`/`dist` after switching branches if you hit `Expected content key … to exist`.

### Caveats

- No linter or test runner is configured.
- Both `yarn.lock` and `package-lock.json` may exist; the deploy path uses `npm` + `git push` to the `gh-pages` branch (site: https://nbardy.github.io/Neural-Force-Field-Art/).
- `tools/smoke.mjs` needs `puppeteer` (a devDependency; downloads a Chromium on install).
