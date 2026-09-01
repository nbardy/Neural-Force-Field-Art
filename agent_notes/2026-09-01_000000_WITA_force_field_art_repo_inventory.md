# Force-field art repo inventory

## Goal

Identify the most recent force-field art directory in `/Users/nicholasbardy/git` and whether multiple copies exist.

## Verified findings

- `Neural-Force-Field-Art` is the active rewrite. Its latest commit is `911ee6b` dated 2026-08-26.
- `force-field-ml-art` is the original project. Its latest commit is `22aafb7` dated 2023-11-29.
- The original README explicitly points to `Neural-Force-Field-Art` as its rewrite.
- Both repos contain force-field artwork/assets; the newer repo has generated artwork under `output/playwright/` and tracked artwork under `docs/assets/`.

## Commands inspected

- `rg --files` for force/field/art-related paths
- `git log -1` and recent `git log` in both repos
- `git ls-tree` for tracked visual assets
- filesystem image inventory sorted by modification time
