# v08 Grid Contact Sheet Prompt

## Hypothesis

The `grid9_close2` CLIP layout already lowers per-view effective resolution by
placing nine rendered camera views into one `256x256` CLIP input, then uses two
full-resolution close-up lanes. The grid lane should get text that matches the
image structure instead of only the normal subject prompt.

This fork makes the grid text explicit and testable:

```text
a 3x3 image grid showing the same subject, {prompt}, from nine different camera
angles, centered on a black background: top-down view, front-facing view, right
side view, rear view, left side view, elevated front-left view looking down,
elevated front-right view looking down, low rear-right view looking up, and low
rear-left view looking up
```

The UI now exposes:

- `grid prompt`: use the contact-sheet prompt above for lane 0.
- `same grid text`: use the normal subject prompt for lane 0.

The two close-up lanes still use the selected per-view prompt mode.

## Snapshot

`snapshot/` contains the 3D camera/prompt, grid layout, optimizer, page, and
benchmark files copied from `HEAD` before this fork was edited.

Diff from snapshot:

```bash
node experiments/clip_forks/diff_fork.mjs v08_grid_contact_sheet_prompt
```

## Commands

Build gate:

```bash
npx parcel build --no-scope-hoist --no-cache src/index.html src/splat.html src/splat3d.html
```

Manual screenshot path:

```bash
yarn start
# Open http://localhost:1234/splat3d.html
# Select "3x3 grid + 2" and toggle "grid prompt" vs "same grid text".
```

Timing should not materially change, because this only changes text embedding
choice and UI state. Quality should be tested by fixed prompt/seed screenshots.

## Result

Build passed:

```text
npx parcel build --no-scope-hoist --no-cache src/index.html src/splat.html src/splat3d.html
✨ Built in 3.26s
```

Headless smoke reached the page, but this environment's headless Chrome reported
`adapter: null`, so visual/WebGPU verification still needs an interactive browser
or a host with a headless WebGPU adapter.
