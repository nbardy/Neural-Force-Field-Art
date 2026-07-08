#!/usr/bin/env bash
# Deploy all three pages (particle art + 2D splats + 3D splats) to GitHub Pages.
#
#   bash tools/splat/deploy_ghpages.sh
#
# Each page is built as its OWN entry with --public-url ./ (relative paths) — a
# COMBINED `parcel build` over the source array emits absolute /bundle.js paths
# that 404 under the /Neural-Force-Field-Art/ subpath. The site fetches the CLIP
# weights from the HF Hub in prod (see src/splat_page.ts), so only HTML+JS ship
# here — no model files. Sourcemaps are excluded to keep the branch lean.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "── building pages (per-entry, relative paths) ──"
for e in index splat splat3d; do
  ./node_modules/.bin/parcel build --no-scope-hoist --no-cache --public-url ./ --dist-dir dist_gh "src/$e.html"
done

echo "── publishing dist_gh → gh-pages ──"
WT="$(mktemp -d)"
git fetch -q origin gh-pages
git worktree add -f -B gh-pages "$WT" origin/gh-pages
git -C "$WT" rm -r --quiet --ignore-unmatch . >/dev/null 2>&1 || true
rsync -a --exclude='*.map' "$ROOT/dist_gh/" "$WT/"
touch "$WT/.nojekyll"   # stop Jekyll from touching the parcel output
git -C "$WT" add -A
git -C "$WT" commit -q -m "deploy: particle art + 2D/3D CLIP splat pages" || echo "(no changes to deploy)"
git -C "$WT" push origin gh-pages
git worktree remove -f "$WT"

echo "── deployed ──"
echo "  https://nbardy.github.io/Neural-Force-Field-Art/           (particle art)"
echo "  https://nbardy.github.io/Neural-Force-Field-Art/splat.html   (2D CLIP splats)"
echo "  https://nbardy.github.io/Neural-Force-Field-Art/splat3d.html (3D CLIP splats)"
