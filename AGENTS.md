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
- **TF.js backend auto-fallback.** `initBackend()` in `main.ts` tries webgpu → webgl → cpu. In headless/GPU-less environments it falls back to CPU, which works but is slower for training. Console shows `TF.js backend: cpu`.
- **Clear `.parcel-cache` after switching branches.** Parcel's cache can produce `Expected content key … to exist` errors when the dependency graph changes. Run `rm -rf .parcel-cache dist` before rebuilding.
- **`draw_webgpu.ts` is non-functional.** It imports `@swissgl/swissgl2` which was never published to npm. The app uses `draw_canvas2d.ts` (Canvas 2D API, works everywhere).
- **Both `yarn.lock` and (sometimes) `package-lock.json` exist.** Use `yarn` as the package manager.
- **ScoreNetwork (trashPanda) model.** `src/trashPanda/models/ScoreNetwork.ts` implements a score function for diffusion-style force fields. Currently configured for the Archimedes spiral demo in `main.ts`.
