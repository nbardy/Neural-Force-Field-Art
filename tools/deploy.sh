#!/usr/bin/env bash
# One command: build → commit main → push main → publish gh-pages → verify live.
#
# Usage:
#   tools/deploy.sh "commit message"   # commits any working-tree changes first
#   tools/deploy.sh                    # tree must already be clean
#
# Two build flags here are load-bearing; changing either ships a blank page:
#   --no-scope-hoist  parcel's default scope-hoisting crashes tfjs at runtime
#                     (ReferenceError: $<hash>$exports is not defined). See AGENTS.md.
#   --public-url ./   GitHub Pages serves this repo from /Neural-Force-Field-Art/,
#                     not from the domain root. Without this parcel emits absolute
#                     "/index.<hash>.js" paths that 404 on the live site while the
#                     HTML itself still returns 200 — a deploy that looks fine to
#                     curl and is broken in a browser. (Hit for real 2026-08-15.)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MESSAGE="${1-}"
STAMP="$(date +%Y%m%d-%H%M%S)"
BUILD_DIR="dist_deploy_${STAMP}"
WORKTREE=".git/tmp-gh-pages-${STAMP}"
PAGES_URL="https://nbardy.github.io/Neural-Force-Field-Art/"

branch="$(git rev-parse --abbrev-ref HEAD)"
if [ "$branch" != "main" ]; then
  echo "deploy: refusing to deploy from '$branch' (expected main)" >&2
  exit 1
fi

# --- 1. commit working tree -------------------------------------------------
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  if [ -z "$MESSAGE" ]; then
    echo "deploy: working tree is dirty and no commit message was given" >&2
    echo "        usage: tools/deploy.sh \"commit message\"" >&2
    exit 1
  fi
  git add -A
  git commit -m "$MESSAGE"
elif [ -n "$MESSAGE" ]; then
  echo "deploy: tree already clean, ignoring commit message"
fi

MAIN_SHA="$(git rev-parse --short HEAD)"
MAIN_SUBJECT="$(git log -1 --pretty=%s)"

# --- 2. build ---------------------------------------------------------------
# Call the local binary directly: `npx`/`npm` are rewritten by a shell hook in
# some agent environments and resolve to the wrong thing.
echo "deploy: building into ${BUILD_DIR}"
./node_modules/.bin/parcel build \
  --no-scope-hoist --no-cache --public-url ./ --dist-dir "$BUILD_DIR" \
  src/index.html src/splat.html src/splat3d.html

for page in index.html splat.html splat3d.html; do
  [ -f "$BUILD_DIR/$page" ] || { echo "deploy: build did not emit $page" >&2; exit 1; }
done
# Guard the failure mode the --public-url comment describes.
if grep -qE '(src|href)="/' "$BUILD_DIR/index.html"; then
  echo "deploy: built index.html has root-absolute asset paths; they will 404 on Pages" >&2
  exit 1
fi

# --- 3. push main -----------------------------------------------------------
git push origin main

# --- 4. publish gh-pages ----------------------------------------------------
git worktree add "$WORKTREE" gh-pages >/dev/null
cleanup() { git worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true; }
trap cleanup EXIT

(
  cd "$WORKTREE"
  git rm -rq .
  cp "$REPO_ROOT/$BUILD_DIR"/*.html "$REPO_ROOT/$BUILD_DIR"/*.js "$REPO_ROOT/$BUILD_DIR"/*.css .
  touch .nojekyll          # keep Pages from running Jekyll over hashed bundles
  git add -A
  if git diff --cached --quiet; then
    echo "deploy: gh-pages already matches this build, nothing to publish"
  else
    git commit -q -m "deploy: ${MAIN_SUBJECT} (main @ ${MAIN_SHA})"
    git push origin gh-pages
  fi
)

# --- 5. verify live ---------------------------------------------------------
# Pages builds asynchronously; a 200 before the build lands is the OLD site.
echo "deploy: waiting for the Pages build"
for _ in $(seq 1 40); do
  status="$(gh api repos/nbardy/Neural-Force-Field-Art/pages/builds/latest --jq .status 2>/dev/null || echo unknown)"
  [ "$status" = "built" ] && break
  sleep 15
done
[ "${status:-}" = "built" ] || { echo "deploy: Pages build did not reach 'built' (last: ${status:-unknown})" >&2; exit 1; }

bundle="$(sed -n 's/.*src="\([^"]*\.js\)".*/\1/p' "$BUILD_DIR/index.html" | head -1)"
for path in "" "$bundle"; do
  code="$(curl -s -o /dev/null -w '%{http_code}' "${PAGES_URL}${path}")"
  [ "$code" = "200" ] || { echo "deploy: ${PAGES_URL}${path} returned $code" >&2; exit 1; }
done
# The bundle name is content-hashed, so serving it proves the NEW build is live.
curl -s "$PAGES_URL" | grep -q "$bundle" \
  || { echo "deploy: live index.html does not reference $bundle" >&2; exit 1; }

echo "deploy: live at $PAGES_URL (main @ ${MAIN_SHA}, bundle ${bundle})"
echo "deploy: headless check → node tools/smoke.mjs $PAGES_URL"
