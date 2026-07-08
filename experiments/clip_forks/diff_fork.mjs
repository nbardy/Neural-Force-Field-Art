#!/usr/bin/env node
import { existsSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { spawnSync } from "node:child_process";

const root = process.cwd();
const fork = process.argv[2];

if (!fork) {
  console.error("usage: node experiments/clip_forks/diff_fork.mjs vNN_name");
  process.exit(2);
}

const snapshotRoot = join(root, "experiments", "clip_forks", fork, "snapshot");
if (!existsSync(snapshotRoot)) {
  console.error(`missing snapshot: ${relative(root, snapshotRoot)}`);
  process.exit(2);
}

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) walk(path, out);
    else out.push(path);
  }
  return out;
}

let hadDiff = false;
for (const before of walk(snapshotRoot)) {
  const rel = relative(snapshotRoot, before);
  const live = join(root, rel);
  if (!existsSync(live)) {
    hadDiff = true;
    console.log(`\n--- ${rel} deleted from live tree ---`);
    continue;
  }
  const res = spawnSync("git", ["diff", "--no-index", "--", before, live], {
    cwd: root,
    encoding: "utf8",
  });
  if (res.status === 1) {
    hadDiff = true;
    process.stdout.write(res.stdout);
    process.stderr.write(res.stderr);
  } else if (res.status && res.status !== 0) {
    process.stdout.write(res.stdout);
    process.stderr.write(res.stderr);
    process.exit(res.status);
  }
}

if (!hadDiff) console.log(`${fork}: no diff against snapshot`);
