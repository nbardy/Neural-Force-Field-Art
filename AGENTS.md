# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

"Halls of Space" / Neural Force Field Art — a single-page React 18 app using TensorFlow.js for neural-network-driven particle simulations rendered on an HTML canvas via WebGL. Bundled with Parcel.

### Dev server

```bash
yarn start          # runs `parcel src/index.html` on port 1234
yarn build          # production build to dist/
```

### Caveats

- **No linter or test runner is configured.** There is no ESLint, Prettier, or test script in `package.json`.
- **GPU required for full rendering.** The app uses TensorFlow.js with WebGPU/WebGL backends. In headless/GPU-less environments the app initialises correctly (console shows "testing max embeddings" and "starting loop") but particles won't render because WebGL context creation fails. This is an environment limitation, not a code bug.
- **Clear `.parcel-cache` after switching branches.** Parcel's cache can produce `Expected content key … to exist` errors when the dependency graph changes. Run `rm -rf .parcel-cache dist` before rebuilding.
- **`draw_webgpu.ts` is non-functional.** It imports `@swissgl/swissgl2` which was never published to npm. `main.ts` uses `draw_webgl.ts` instead.
- **Both `yarn.lock` and (sometimes) `package-lock.json` exist.** Use `yarn` as the package manager.
