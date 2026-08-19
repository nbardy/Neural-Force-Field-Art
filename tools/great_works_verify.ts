/**
 * Make GREAT_WORKS.md falsifiable.
 *
 *   bun tools/great_works_verify.ts
 *
 * A saved recipe is worthless if it silently stops loading. Every entry in
 * GREAT_WORKS.md carries a settings JSON and a `?dock=` link, and this asserts
 * the four ways that pairing can rot:
 *
 *   1. link ≠ json      — someone edited one and not the other
 *   2. link won't decode — transport damage (a `+` that became a space, etc.)
 *   3. pieceName gone    — the piece was renamed out from under the recipe
 *   4. commit gone       — the recorded SHA is not in this repo
 *
 * Needs no GPU and no DOM: it uses the same src/share.ts codec the dock uses,
 * and reads GALLERY names straight out of main.ts by regex.
 */
import { readFileSync } from "fs";
import { execSync } from "child_process";
import { encodeDockParam, decodeDockParam, resolveSharedPiece } from "../src/share";

const DOC = "GREAT_WORKS.md";
const doc = readFileSync(DOC, "utf8");

const main = readFileSync("src/main.ts", "utf8");
const galleryNames = [
  ...main.slice(main.indexOf("export const GALLERY")).matchAll(/^    name: "(.+?)",/gm),
].map((m) => m[1]);
if (galleryNames.length === 0) throw new Error("could not read GALLERY names from src/main.ts");

const jsonBlocks = [...doc.matchAll(/```json\n([\s\S]*?)```/g)].map((m) => m[1]);
const dockParams = [...doc.matchAll(/[?&]dock=([A-Za-z0-9_-]+)/g)].map((m) => m[1]);
const shas = [...doc.matchAll(/^-\s+\*\*Commit:\*\*\s+`([0-9a-f]{7,40})`/gm)].map((m) => m[1]);

let failures = 0;
const ok = (cond: boolean, msg: string) => {
  console.log(`${cond ? "  ok  " : " FAIL "} ${msg}`);
  if (!cond) failures++;
};

console.log(`=== ${DOC}: ${jsonBlocks.length} entr(y|ies) ===\n`);
ok(
  jsonBlocks.length > 0 && jsonBlocks.length === dockParams.length,
  `${jsonBlocks.length} json block(s) pair with ${dockParams.length} ?dock= link(s)`
);

jsonBlocks.forEach((raw, i) => {
  let dock: any;
  try {
    dock = JSON.parse(raw);
  } catch (e) {
    ok(false, `entry ${i}: settings JSON does not parse (${(e as Error).message})`);
    return;
  }
  const label = dock.pieceName ?? `entry ${i}`;

  // 1 + 2: the link IS the json, both directions.
  const param = dockParams[i];
  ok(param === encodeDockParam(dock), `${label}: ?dock= link matches the recorded JSON`);
  const decoded = decodeDockParam(param ?? "");
  ok(decoded.tag === "ok", `${label}: ?dock= link decodes`);
  if (decoded.tag === "ok") {
    ok(
      JSON.stringify(decoded.json) === JSON.stringify(dock),
      `${label}: decoded link is byte-identical to the JSON`
    );
  }

  // 3: the recipe still names a piece this build has.
  const piece = resolveSharedPiece(galleryNames, dock.runtime?.piece, dock.pieceName);
  ok(
    piece.tag !== "unknown-name",
    `${label}: piece resolves (${piece.tag}${
      piece.tag === "renamed" ? ` — index moved ${piece.staleIndex}→${piece.piece}` : ""
    })`
  );
});

// 4: every recorded commit is still reachable.
shas.forEach((sha) => {
  let exists = true;
  try {
    execSync(`git cat-file -e ${sha}^{commit}`, { stdio: "ignore" });
  } catch {
    exists = false;
  }
  ok(exists, `commit ${sha} is in this repo`);
});

console.log(failures === 0 ? "\nALL PASS" : `\n${failures} FAILURE(S)`);
process.exit(failures === 0 ? 0 : 1);
